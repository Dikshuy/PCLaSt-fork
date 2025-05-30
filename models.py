"""
Optimized neural network models for hierarchical RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
import mixer


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for encoding continuous values"""
    
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self._const = 2 * np.pi
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * self._const
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLP(nn.Module):
    """Residual MLP block with layer normalization"""
    
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden)
        )
    
    def forward(self, x):
        return x + self.net(x)


class AC(nn.Module):
    """Action prediction network with k-step embedding"""
    
    def __init__(self, din, nk, nact):
        super().__init__()
        self.din = din
        self.nact = nact
        
        # Embeddings
        self.k_embedding = nn.Embedding(nk, din)
        self.action_embedding = nn.Embedding(150, din)
        
        # Main network with residual blocks
        self.network = nn.Sequential(
            nn.Linear(din * 3, 256),
            ResidualMLP(256),
            ResidualMLP(256)
        )
        
        # Output heads
        self.continuous_head = nn.Linear(256, nact)
        self.discrete_head = nn.Linear(256, 150)
        
        # Optional components
        self.y_encoder = nn.Sequential(
            nn.Linear(2, din),
            nn.Tanh(),
            nn.Linear(din, din)
        )
        self.time_encoder = GaussianFourierProjection(din // 2)
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, st, stk, k, a):
        """Forward pass computing action prediction loss"""
        a_continuous = a[:, :2]
        a_discrete = a[:, 2].long()
        
        # Embed k-step
        k_embed = self.k_embedding(k)
        
        # Concatenate features
        features = torch.cat([st, stk, k_embed], dim=1)
        
        # Main network
        h = self.network(features)
        
        # Predictions
        cont_pred = self.continuous_head(h)
        disc_pred = self.discrete_head(h)
        
        # Compute losses
        continuous_loss = F.mse_loss(cont_pred, a_continuous) * 10.0
        discrete_loss = self.ce_loss(disc_pred, a_discrete) * 0.01
        
        return continuous_loss + discrete_loss


class LatentForward(nn.Module):
    """Latent forward dynamics model with optional VAE components"""
    
    def __init__(self, dim, nact, use_vae=True):
        super().__init__()
        self.dim = dim
        self.nact = nact
        self.use_vae = use_vae
        
        # Main dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(dim + nact, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, dim)
        )
        
        if use_vae:
            # VAE components
            self.prior_net = nn.Sequential(
                nn.Linear(dim + nact, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512)
            )
            
            self.posterior_net = nn.Sequential(
                nn.Linear(dim * 2 + nact, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512)
            )
            
            self.decoder_net = nn.Sequential(
                nn.Linear(dim + nact + 256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, dim)
            )
    
    def forward(self, z, a, detach=True):
        """Forward dynamics prediction"""
        a = a[:, :2]  # Use only continuous actions
        
        if detach:
            z = z.detach()
        
        z_action = torch.cat([z, a], dim=1)
        
        if self.use_vae:
            # VAE forward pass
            prior_params = self.prior_net(z_action)
            mu_prior, log_std_prior = torch.chunk(prior_params, 2, dim=1)
            std_prior = torch.exp(log_std_prior) * 0.001 + 1e-5
            
            prior_dist = torch.distributions.Normal(mu_prior, std_prior)
            sample = prior_dist.rsample()
            
            z_pred = self.decoder_net(torch.cat([z_action, sample], dim=1))
        else:
            z_pred = self.dynamics_net(z_action)
        
        return z_pred
    
    def loss(self, z, zn, a, do_detach=True):
        """Compute forward dynamics loss with optional VAE components"""
        a = a[:, :2]
        
        if do_detach:
            z = z.detach()
            zn = zn.detach()
        
        z_action = torch.cat([z, a], dim=1)
        
        if self.use_vae:
            # Prior distribution
            prior_params = self.prior_net(z_action)
            mu_prior, log_std_prior = torch.chunk(prior_params, 2, dim=1)
            std_prior = torch.exp(log_std_prior)
            prior_dist = torch.distributions.Normal(mu_prior, std_prior)
            
            # Posterior distribution
            posterior_input = torch.cat([z_action, zn], dim=1)
            posterior_params = self.posterior_net(posterior_input)
            mu_posterior, log_std_posterior = torch.chunk(posterior_params, 2, dim=1)
            std_posterior = torch.exp(log_std_posterior)
            posterior_dist = torch.distributions.Normal(mu_posterior, std_posterior)
            
            # KL divergence
            kl_loss = torch.distributions.kl_divergence(
                posterior_dist, prior_dist
            ).sum(dim=-1).mean()
            
            # Sample and decode
            sample = posterior_dist.rsample()
            z_pred = self.decoder_net(torch.cat([z_action, sample], dim=1))
            
            # Reconstruction loss
            recon_loss = F.mse_loss(z_pred, zn) * 0.1
            
            return recon_loss + kl_loss * 0.01, z_pred
        else:
            z_pred = self.dynamics_net(z_action)
            loss = F.mse_loss(z_pred, zn) * 0.1
            return loss, z_pred


class Encoder(nn.Module):
    """Image encoder with contrastive learning components"""
    
    def __init__(self, din, dout, ndiscrete=64):
        super().__init__()
        self.din = din
        self.dout = dout
        self.ndiscrete = ndiscrete
        
        # Mixer network for feature extraction
        self.mixer = mixer.MLP_Mixer(
            n_layers=2, n_channel=32, n_hidden=32, 
            n_output=32*4*4, image_size_h=100, 
            image_size_w=100, patch_size=10, n_image_channel=1
        )
        
        # Normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.GroupNorm(4, 32)
        
        # Main encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(32*4*4*2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, dout)
        )
        
        # Contrastive learning components
        self.contrastive_net = nn.Sequential(
            nn.Linear(dout, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, dout)
        )
        
        # Contrastive parameters
        self.w_contrast = nn.Linear(1, 1)
        self.b_contrast = nn.Linear(1, 1)
        
        # Inverse network for contrastive learning
        self.contrast_inverse = nn.Sequential(
            nn.Linear(dout, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, dout)
        )
        
        # Vector quantization
        self.vq = VectorQuantize(
            dim=dout, 
            codebook_size=ndiscrete, 
            decay=0.8, 
            commitment_weight=1.0, 
            threshold_ema_dead_code=0.1,
            heads=1, 
            kmeans_init=True
        )
    
    def forward(self, x):
        """Encode image to latent representation"""
        # Reshape input
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 100, 100)
        
        # Extract features with mixer
        h = self.mixer(x)
        
        # Apply normalization
        h1 = self.bn1(h)
        h = h.view(batch_size, 32, 4, 4)
        h2 = self.bn2(h)
        
        # Concatenate normalized features
        h1 = h1.view(batch_size, -1)
        h2 = h2.view(batch_size, -1)
        h = torch.cat([h1, h2], dim=1)
        
        # Final encoding
        return self.encoder_net(h)
    
    def contrastive(self, z):
        """Apply contrastive projection"""
        return self.contrastive_net(z)
    
    def contrast_inv(self, z):
        """Inverse mapping for contrastive space"""
        return self.contrast_inverse(z)


class Probe(nn.Module):
    """Probe network for grounding latent states"""
    
    def __init__(self, din, dout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(din, 256),
            ResidualMLP(256),
            nn.Linear(256, dout)
        )
    
    def forward(self, s):
        """Forward pass"""
        return self.encoder(s)
    
    def loss(self, s, ground_truth):
        """Compute probing loss"""
        # Detach to prevent gradients flowing back
        s_detached = s.detach()
        predictions = self.encoder(s_detached)
        
        # MSE loss
        mse_loss = F.mse_loss(predictions, ground_truth)
        
        # Absolute error for logging
        abs_error = torch.abs(predictions - ground_truth).mean()
        
        return mse_loss, abs_error


class DistPred(nn.Module):
    """Distance prediction network"""
    
    def __init__(self, latent_dim, max_k):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_k = max_k
        
        # Network for distance prediction
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, z1, z2):
        """Predict distance between two latent states"""
        combined = torch.cat([z1, z2], dim=1)
        return self.network(combined)
    
    def predict_k(self, z1, z2):
        """Predict k-step distance"""
        return self.forward(z1, z2).squeeze(-1)
    
    def loss(self, z1, z2, k):
        """Compute distance prediction loss"""
        pred_k = self.forward(z1, z2).squeeze(-1)
        return F.mse_loss(pred_k, k.float())
