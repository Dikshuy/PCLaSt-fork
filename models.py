import torch
import torch.nn as nn
import mixer
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

import random

#from quantize import EMAVectorQuantizer, VectorQuantizer2
from vector_quantize_pytorch import VectorQuantize

# from batchrenorm import BatchRenorm1d

ce = nn.CrossEntropyLoss()


class GaussianFourierProjectionTime(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResMLP(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_hidden)
        self.mlp3 = nn.Linear(n_hidden, n_hidden)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_hidden)

    def forward(self, U):
        z = self.layer_norm(U)  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))  # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)  # z: [B, n_tokens, n_channel]
        Y = U + z  # Y: [B, n_tokens, n_channel]
        return Y


class AC(nn.Module):
    def __init__(self, din, nk, nact):
        super().__init__()

        self.kemb = nn.Embedding(nk, din)
        self.cemb = nn.Embedding(150, din)

        #self.m = nn.Sequential(nn.Linear(din*3, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.m = nn.Sequential(nn.Linear(din*3, 256), ResMLP(256), ResMLP(256))

        #self.m_cond = nn.Sequential(nn.Linear(din*4, 256), ResMLP(256), ResMLP(256))

        self.acont_pred = nn.Linear(256, nact)
        self.adisc_pred = nn.Linear(256, 150)

        self.yl = nn.Sequential(nn.Linear(2,din), nn.Tanh(), nn.Linear(din,din))
        self.tl = GaussianFourierProjectionTime(din//2)

    def forward(self, st, stk, k, a): 

        loss = 0.0

        a_cont = a[:,:2]
        a_disc = a[:,2].long()


        #noise_t = torch.rand_like(a[:,0])
        #noised_a,_,_ = noised(a, noise_t)
        #yemb = self.yl(noised_a)
        #temb = self.tl(noise_t)

        ke = self.kemb(k)

        h = torch.cat([st, stk, ke], dim=1)

        #if self.training:
        #    mix_alpha = 0.2
        #    lam = np.random.beta(mix_alpha+1, mix_alpha)
        #    h = lam*h + (1-lam)*h[torch.randperm(h.shape[0])]

        h = self.m(h)

        acont_p = self.acont_pred(h)
        adisc_p = self.adisc_pred(h)

        loss += ((acont_p - a_cont)**2).sum(dim=-1).mean() * 10.0

        loss += 0.01 * ce(adisc_p, a_disc)

        return loss


class LatentForward(nn.Module):
    def __init__(self, dim, nact):
        super().__init__()
        self.alin = nn.Sequential(nn.Linear(nact, dim))
        self.net = nn.Sequential(nn.Linear(dim + nact, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, dim))
        #self.net = nn.Sequential(ResMLP(512), ResMLP(512), nn.Linear(512, dim))

        if True:
            self.prior = nn.Sequential(nn.Linear(dim + nact, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,512))
            self.posterior = nn.Sequential(nn.Linear(dim*2 + nact, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512,512))
            self.decoder = nn.Sequential(nn.Linear(dim + nact + 256, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512,dim))

    # zn = forward(z, a)
    # grad(zn, z) # grad of scalar-output wrt vector-input.  
    # ILQR get jacobian which is (mxm).  

    def forward(self, z, a, detach = True):
        a = a[:,:2]
        zc = torch.cat([z.detach() if detach else z, a], dim=1)

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(zc), 2, dim=1)
            std_prior = torch.exp(std_prior) * 0.001 + 1e-5
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            sample = prior.rsample()
            zpred = self.decoder(torch.cat([zc, sample], dim=1))
        else:
            zpred = self.net(zc)

        return zpred

    def loss(self, z, zn, a, do_detach=True):
        a = a[:,:2]

        if do_detach:
            detach_fn = lambda inp: inp.detach()
        else:
            detach_fn = lambda inp: inp

        zc = torch.cat([z,a], dim=1)
        zpred = self.net(zc)

        loss = 0.0

        if True:
            mu_prior, std_prior = torch.chunk(self.prior(detach_fn(zc)), 2, dim=1)
            mu_posterior, std_posterior = torch.chunk(self.posterior(torch.cat([detach_fn(zc), detach_fn(zn)], dim=1)), 2, dim=1)
            std_prior = torch.exp(std_prior)
            std_posterior = torch.exp(std_posterior)
            prior = torch.distributions.normal.Normal(mu_prior, std_prior)
            posterior = torch.distributions.normal.Normal(mu_posterior, std_posterior)
            kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()
            sample = posterior.rsample()
            zpred = self.decoder(torch.cat([detach_fn(zc), sample], dim=1))
            zn = detach_fn(zn)
            loss += kl_loss * 0.01
        else:
            zpred = self.forward(z, a)

        loss += ((zn - zpred)**2).sum(dim=-1).mean() * 0.1

        return loss, zpred


class Encoder(nn.Module):
    def __init__(self, din, dout, ndiscrete=None):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        contrastive_dim = dout
        self.mixer = mixer.MLP_Mixer(n_layers=2, n_channel=32, n_hidden=32, n_output=32*4*4, image_size_h=100, image_size_w=100, patch_size=10, n_image_channel=1)

        #self.m = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256,256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Linear(256, dout))
        #self.m = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256,256), nn.LeakyReLU(), nn.Linear(256, dout))

        self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm2d(32)
        self.bn2 = nn.GroupNorm(4,32)

        self.m = nn.Sequential(nn.Linear(32*4*4*2, 256), nn.LeakyReLU(), nn.Linear(256, dout))

        self.contrastive = nn.Sequential(nn.Linear(dout, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, contrastive_dim)).to(device)
        self.w_contrast = nn.Linear(1,1).to(device)#nn.Parameter(torch.ones(1,1).float()).cuda()
        self.b_contrast = nn.Linear(1,1).to(device)#nn.Parameter(torch.ones(1,1).float()).cuda()
        self.contrast_inv = nn.Sequential(nn.Linear(contrastive_dim, 512), nn.LeakyReLU(), nn.Linear(512, dout)).to(device)
        self.vq = VectorQuantize(dim=contrastive_dim, codebook_size=ndiscrete, decay=0.8, commitment_weight=1.0, threshold_ema_dead_code = 0.1, heads=1, kmeans_init=False).to(device)

    def forward(self, x):

        x = x.reshape((x.shape[0], 1, 100, 100))

        #p1 = torch.arange(0,1,0.01).reshape((1, 1, 100, 1)).repeat((x.shape[0], 1, 1, 100)).cuda()
        #p2 = torch.arange(0,1,0.01).reshape((1, 1, 1, 100)).repeat((x.shape[0], 1, 100, 1)).cuda()
        #x = torch.cat([x,p1,p2],dim=1)
        
        h = self.mixer(x)

        h1 = self.bn1(h)
        h = h.reshape((h.shape[0], 32, 4, 4))
        h2 = self.bn2(h)

        h1 = h1.reshape((h.shape[0], -1))
        h2 = h2.reshape((h.shape[0], -1))

        h = torch.cat([h1*0.0,h2],dim=1)

        return self.m(h)

    def to(self, device):
        self.m = self.m.to(device)
        self.mixer = self.mixer.to(device)
        self.bn1 = self.bn1.to(device)
        self.bn2 = self.bn2.to(device)
        return self


class Probe(nn.Module):
    def __init__(self, din, dout):
        super().__init__()

        #self.enc = nn.Sequential(nn.Linear(din, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512,512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Linear(512, dout))

        self.enc = nn.Sequential(nn.Linear(din, 256), ResMLP(256), nn.Linear(256, dout))
        #self.enc = nn.Linear(din, dout)

    def forward(self, s):
        return self.enc(s)

    def loss(self, s, gt):
        # print('input', s[:10])
        # print('target', gt[:10])
        # print('input-target diff', (torch.abs(s - gt)).sum())

        # print('in probe')
        # print(s[0])

        sd = s.detach()

        sd = self.enc(sd)

        loss = ((sd - gt) ** 2).sum(dim=-1).mean()

        abs_loss = torch.abs(sd - gt).mean()

        return loss, abs_loss