from room_env import RoomEnv
import os, time
from os.path import join
from datetime import datetime
from room_obstacle_env import RoomObstacleEnv
from room_polygon_obstacle_env import RoomPolygonObstacleEnv,RoomMultiPassageEnv ,RoomSpiral
from room_polygon_obstacle_env import RoomMultiPassageEnvLarge
import matplotlib
import random

from dist_pred_model import DistPred

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from models import Encoder, Probe, AC, LatentForward
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import wandb
from emprical_mdp import EmpiricalMDP
from ema_pytorch import EMA
import os
import copy
import time
import torch.nn as nn
from dijkstra import make_ls, DP_goals

'''
Sample 100k examples.  
Write a batch sampler to get (xt, xtk, k, agent_state, block_state).  

p(a[t] | enc(x[t]), enc(x[t+k]), k)

enc(x[t]).detach() --> agent_state[t]
enc(x[t]).detach() --> block_state[t]

encoder
inv_model
as_probe
bs_probe


'''


def abstract_path_sampler(empirical_mdp, abstract_horizon):
    plan = {'states': [], 'actions': []}

    init_state = np.random.choice(empirical_mdp.unique_states)
    plan['states'].append(init_state)

    while len(plan['states']) != abstract_horizon:
        current_state = plan['states'][-1]
        next_state_candidates = []
        for state in empirical_mdp.unique_states:
            if not np.isnan(empirical_mdp.transition[current_state][state]).all() and state != current_state:
                next_state_candidates.append(state)
        next_state = np.random.choice(next_state_candidates)
        plan['actions'].append(empirical_mdp.transition[current_state, next_state])
        plan['states'].append(next_state)

    return plan


def obs_sampler(dataset_obs, dataset_agent_states, state_labels, abstract_state):
    # np.random.seed(0)
    _filtered_obs = dataset_obs[state_labels == abstract_state]
    _filtered_agent_states = dataset_agent_states[state_labels == abstract_state]
    index = np.random.choice(range(len(_filtered_obs)))
    return _filtered_obs[index], _filtered_agent_states[index]


def sample_example(X, A, ast, est, max_k):
    N = X.shape[0]
    t = random.randint(0, N - max_k - 1)
    k = random.randint(1, max_k)

    return (X[t], X[t + 1], X[t + k], k, A[t], ast[t], est[t])


def sample_batch(X, A, ast, est, bs, max_k):
    xt = []
    xtn = []
    xtk = []
    klst = []
    astate = []
    estate = []
    alst = []

    for b in range(bs):
        lst = sample_example(X, A, ast, est, max_k=max_k)
        xt.append(lst[0])
        xtn.append(lst[1])
        xtk.append(lst[2])
        klst.append(lst[3])
        alst.append(lst[4])
        astate.append(lst[5])
        estate.append(lst[6])

    xt = torch.Tensor(np.array(xt)).to(device)
    xtn = torch.Tensor(np.array(xtn)).to(device)
    xtk = torch.Tensor(np.array(xtk)).to(device)
    klst = torch.Tensor(np.array(klst)).long().to(device)
    alst = torch.Tensor(np.array(alst)).to(device)
    astate = torch.Tensor(np.array(astate)).to(device)
    estate = torch.Tensor(np.array(estate)).to(device)

    return xt, xtn, xtk, klst, alst, astate, estate


class LatentWrapper(nn.Module):
    def __init__(self, latent) -> None:
        super().__init__()
        self.latent = latent
        self.nu = 2

    def forward(self, z, a):
        return self.latent(z, a, detach=False)


class Cluster_Transform:
    def __init__(self, enc, fwd_dyn, device='cuda') -> None:
        self.enc = enc
        self.fwd_dyn = fwd_dyn
        self.device = device

    def cluster_label_transform(self, X, do_augment = False, contrastive = True):
        # generate label [z, f(z, a_1), f(z, a_2), f(z, a_3), f(z, a_4)] from observations X for clustering
        # with z = enc(X) being the latent state and f(.,.) being the latent forward dynamics
        img = torch.FloatTensor(X).to(self.device)
        z = self.enc(img)

        actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(self.device)
        
        if not contrastive:
            output = z
        else:
            #output,_,_ = self.enc.vq(self.enc.contrastive(z).unsqueeze(1))
            #output = output.squeeze(1)
            output = self.enc.contrastive(z)

        if do_augment:
            for i in range(4):
                y = self.fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))

                if contrastive:
                    y = self.enc.contrastive(y)

                output = torch.cat((output, y.detach()), axis=1)

        return output

    def cluster_label_transform_latent(self, z, do_augment = False, contrastive = True):
        # augment the latent state z for clustering
        actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(self.device)

        if contrastive:
            output,_,_ = self.enc.vq(self.enc.contrastive(z).unsqueeze(1))
            output = output.squeeze(1)
        
        if do_augment:
            for i in range(4):
                y = self.fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))
                
                if contrastive:
                        y = self.enc.contrastive(y)
                                    
                output = torch.cat((output, y.detach()), axis=1)

        return output


# def cluster_label_transform(X, enc, fwd_dyn, device = 'cuda'):
#     # generate label [z, f(z, a_1), f(z, a_2), f(z, a_3), f(z, a_4)] from observations X for clustering
#     # with z = enc(X) being the latent state and f(.,.) being the latent forward dynamics
#     img = torch.FloatTensor(X).to(device)
#     z = enc(img)

#     actions = torch.FloatTensor([[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]).to(device)
#     output = z
#     for i in range(4):
#         y = fwd_dyn(z, actions[i].unsqueeze(0).repeat(z.size(0), 1))
#         output = torch.cat((output, y.detach()), axis = 1)

#     return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name', default='acsnc',
                            help='name of the wandb project')
    wandb_args.add_argument('--use-wandb', action='store_true',
                            help='use Weight and bias visualization lib')
    # training args
    train_args = parser.add_argument_group('wandb setup')
    train_args.add_argument("--opr", default="high-low-plan",
                        choices=['generate-data', 'train', 'cluster-latent', 'generate-mdp', 'high-low-plan', 'evaluate-planners'])
    train_args.add_argument("--latent-dim", default=256, type=int)
    train_args.add_argument("--num-data-samples", default=500000, type=int)
    train_args.add_argument("--k_embedding_dim", default=45, type=int)
    train_args.add_argument("--max_k", default=2, type=int)
    train_args.add_argument("--do-mixup", action='store_true', default=False)
    train_args.add_argument("--dist-learn", action='store_true', default=False)
    train_args.add_argument("--contrastive", action='store_true', default=False)

    train_args.add_argument("--batch_size", default=128, type=int)

    train_args.add_argument("--contrastive_k", default=1, type=int)
    train_args.add_argument("--ndiscrete", default=64, type=int)
    train_args.add_argument("--nclusters", default=16, type=int)

    train_args.add_argument("--env", default='polygon-obs', 
                            choices=['rat', 'room', 'obstacle', 'polygon-obs',
                                     'room-multi-passage', 'room-spiral',
                                     'room-multi-passage-large'])

    train_args.add_argument('--logdir', default='logs', type=str)
    train_args.add_argument('--exp_id', default='test', type=str)
    train_args.add_argument('--from_to', default=0, nargs="+", type=int)
    train_args.add_argument('--scaling_factor', default=1.0, type=float)
    train_args.add_argument("--seed", default=0, type=int)
    wandb_args.add_argument('--use-augmented-latent-clustering', action='store_true',
                            help='uses augmented latent states for clustering')

    # process arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # wandb init
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest]
                             for x in train_args._group_actions})

    # Train
    if args.env == 'rat':
        from rat_env import RatEnvWrapper

        env = RatEnvWrapper()
    elif args.env == 'room':
        env = RoomEnv()
    elif args.env == 'obstacle':
        env = RoomObstacleEnv()
    elif args.env == 'polygon-obs':
        env = RoomPolygonObstacleEnv()
    elif args.env == 'room-multi-passage':
        env = RoomMultiPassageEnv()
    elif args.env == 'room-spiral':
        env = RoomSpiral()
    elif args.env == 'room-multi-passage-large':
        env = RoomMultiPassageEnvLarge()

    ac = AC(din=args.latent_dim, nk=args.k_embedding_dim, nact=2).to(device)
    enc = Encoder(100 * 100, args.latent_dim, args.ndiscrete).to(device)
    forward = LatentForward(args.latent_dim, 2).to(device)
    a_probe = Probe(args.latent_dim, 2).to(device)
    b_probe = Probe(args.latent_dim, 2).to(device)
    e_probe = Probe(args.latent_dim, 2).to(device)
    dist_pred = DistPred(args.latent_dim, 2000).to(device)
    ema_enc = EMA(enc, beta=0.99)
    ema_forward = EMA(forward, beta=0.99)
    ema_a_probe = EMA(a_probe.enc, beta=0.99)

    logdir = os.path.expanduser(args.logdir)
    logdir = os.path.join(logdir, f"{args.env}")
    os.makedirs(logdir, exist_ok=True)

    field_folder = os.path.join(logdir, "fields")  # + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')
    plan_folder = os.path.join(logdir, "fields")  # + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')
    data_folder = os.path.join(logdir, 'data')
    dataset_path = os.path.join(data_folder, 'dataset.p')
    model_path = os.path.join(data_folder, 'model.p')
    os.makedirs(data_folder, exist_ok=True)

    train_data_folder = os.path.join(logdir, 'train_data')
    train_data_path = os.path.join(logdir, 'train_data', 'loss_log.p')

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    os.makedirs(field_folder, exist_ok=True)
    os.makedirs(plan_folder, exist_ok=True)
    os.makedirs(train_data_folder, exist_ok=True)

    if args.opr == 'generate-data':
        X = []
        A = []
        ast = []
        est = []

        for i in tqdm(range(0, args.num_data_samples)):
            a = env.random_action()

            x, agent_state, exo_state = env.get_obs()
            env.step(a)

            A.append(a[:])
            X.append(x[:])
            ast.append(agent_state[:])
            est.append(exo_state[:])

        X = np.asarray(X).astype('float32')
        A = np.asarray(A).astype('float32')
        ast = np.array(ast).astype('float32')
        est = np.array(est).astype('float32')

        pickle.dump({'X': X, 'A': A, 'ast': ast, 'est': est}, open(dataset_path, 'wb'))
        pickle.dump({'X': X}, open(os.path.join(data_folder, "images.p"), 'wb'))
        pickle.dump({'A': A}, open(os.path.join(data_folder, "actions.p"), 'wb'))
        pickle.dump({'ast': ast}, open(os.path.join(data_folder, "agent_state.p"), 'wb'))
        pickle.dump({'est': est}, open(os.path.join(data_folder, "exo_state.p"), 'wb'))

        print(f'data generated and stored in {dataset_path}')
    elif args.opr == 'train':
        
        dataset = pickle.load(open(dataset_path, 'rb'))
        print(f"Loading dataset from {dataset_path}")
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
        opt = torch.optim.Adam(list(ac.parameters())
                               + list(enc.parameters())
                               + list(a_probe.parameters())
                               + list(b_probe.parameters())
                               + list(forward.parameters())
                               + list(dist_pred.parameters()), lr=0.0001)

        # plt.figure()
        # for obstacle in env.obs_lst:
        #     x, y = obstacle.exterior.xy
        #     plt.plot(x, y, 'k-')

        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('x')
        # plt.ylabel('y')
        
        # # sample trajectory
        # traj = dataset['ast']
        # length = 10000
        # start = 200000
        # sample_traj = traj[start:start+length, :]
        # plt.plot(sample_traj[:,0], sample_traj[:, 1], 'b-', linewidth = 0.1)
        # plt.scatter(sample_traj[:,0], sample_traj[:, 1], s = 1, c='b')
        # plt.savefig('env_plot_tunnel.png')

        colors = iter(plt.cm.inferno_r(np.linspace(.25, 1, 200000)))
        print('Num samples', X.shape[0])

        print('Run K-mneas')
        kmeans = KMeans(n_clusters=20, verbose=1).fit(A)
        print(' K-Means done')

        A = np.concatenate([A, kmeans.labels_.reshape((A.shape[0], 1))], axis=1)

        loss_log = {'AC_loss':[], 'A_loss':[], 'Asqr_loss':[],
                    'Z_loss':[], 'contrast_loss':[], 'contrast_acc':[]}

        for j in tqdm(range(0, 200000)):
            ac.train()
            enc.train()
            a_probe.train()
            forward.train()
            xt, xtn, xtk, k, a, astate, estate = sample_batch(X, A, ast, est, args.batch_size, max_k=args.max_k)
            astate = torch.round(astate, decimals=3)

            # print('-----')

            # xjoin = torch.cat([xt,xtn,xtk],dim=0)
            # sjoin = enc(xjoin)
            # st, stn, stk = torch.chunk(sjoin, 3, dim=0)

            # do_bn = (j < 5000)

            st = enc(xt)
            stk = enc(xtk)

            stn = enc(xtn)

            ac_loss = ac(st, stk, k, a)
            ap_loss, ap_abserr = a_probe.loss(st, astate)
            ep_loss = ap_loss * 0.0

            z_loss, z_pred = forward.loss(st, stn, a)

            # raise Exception()

            loss = 0

            if args.do_mixup:
                # add probing loss in mixed hidden states.
                mix_lamb = np.random.beta(0.5,0.5)#random.uniform(0, 1)
                mix_ind = torch.randperm(st.shape[0])

                st_mix = st * mix_lamb + st[mix_ind] * (1 - mix_lamb)
                # astate_mix = astate*mix_lamb + astate[mix_ind]*(1-mix_lamb)
                # ap_loss_mix, _ = a_probe.loss(st_mix, astate_mix)
                # loss += ap_loss_mix

                stn_mix = stn * mix_lamb + stn[mix_ind] * (1 - mix_lamb)
                z_loss_mix, _ = forward.loss(st_mix, stn_mix, a, do_detach=False)

                loss += z_loss_mix

            if args.contrastive:
                xt_d, xtn_pos, xt_pos, _, _, _, _ = sample_batch(X, A, ast, est, 128, max_k=args.contrastive_k) #take pos from 1/2 steps away.  
                xt_neg, _, _, _, _, _, _ = sample_batch(X, A, ast, est, 128, max_k=1)

                contrast_loss = 0.0

                zt_d = enc(xt_d).detach()
                zt_pos = enc(xt_pos).detach()
                zt_neg = enc(xt_neg).detach()

                spectral_version = False

                if not spectral_version:
                    vqop = True
                    if vqop:
                        st_d, _, vql1 = enc.vq(enc.contrastive(zt_d).unsqueeze(1))
                        st_pos, _, vql2 = enc.vq(enc.contrastive(zt_pos).unsqueeze(1))
                        st_neg, _, vql3 = enc.vq(enc.contrastive(zt_neg).unsqueeze(1))
                        st_d = st_d.squeeze(1)
                        st_pos = st_pos.squeeze(1)
                        st_neg = st_neg.squeeze(1)
                        contrast_loss += (vql1 + vql2 + vql3).mean()
                    else:
                        st_d = enc.contrastive(zt_d)
                        st_pos = enc.contrastive(zt_pos)
                        st_neg = enc.contrastive(zt_neg)

                    st_d = st_d.squeeze(1)
                    st_pos = st_pos.squeeze(1)
                    st_neg = st_neg.squeeze(1)


                    st_combined = torch.cat([st_d, st_neg],dim=0)
                    st_bs, st_dim = st_combined.shape
                    st_v1 = st_combined.reshape((st_bs, 1, st_dim))
                    st_v2 = st_combined.reshape((1, st_bs, st_dim))

                    neg_dist = torch.sqrt(torch.sum((st_v1 - st_v2)**2, dim=-1))

                    neg_dist = neg_dist.flatten().unsqueeze(-1)

                    pos_dist = torch.sqrt(((st_d - st_pos)**2).sum(dim=1,keepdim=True))
                    neg_dist = torch.sqrt(((st_d - st_neg)**2).sum(dim=1,keepdim=True))

                    pos = torch.exp(enc.b_contrast.bias) - torch.exp(enc.w_contrast.weight) * pos_dist
                    neg = torch.exp(enc.b_contrast.bias) - torch.exp(enc.w_contrast.weight) * neg_dist

                    #pos = 5.0 - pos_dist
                    #neg = 5.0 - neg_dist

                    if random.uniform(0,1) < 0.005:
                        print('pos-dist', pos_dist.shape, pos_dist[1:8])
                        print('pos', pos.shape, pos[1:8])
                        print('neg-dist', neg_dist.shape, neg_dist[1:8])
                        print('neg', neg.shape, neg[1:8])

                        print('xdiffs', 'pos', ((xt_d - xt_pos)**2).mean(), 'neg', ((xt_d - xt_neg)**2).mean())
                        print('zdiffs', 'pos', ((zt_d - zt_pos)**2).mean(), 'neg', ((zt_d - zt_neg)**2).mean())
                        print('sb0,sb1 diff', 'st_d', ((st_d[0]-st_d[1])**2).mean(), 'st_pos', ((st_pos[0]-st_pos[1])**2).mean())
                        print('zb0,zb1 diff', 'zt_d', ((zt_d[0]-zt_d[1])**2).mean(), 'zt_pos', ((zt_pos[0]-zt_pos[1])**2).mean())

                    bce = nn.BCEWithLogitsLoss()

                    contrast_loss += bce(pos, torch.ones_like(pos)).mean()

                    #neg = neg.reshape((st_bs, st_bs))
                    #neg.fill_diagonal_(-2.0)
                    #neg = neg.reshape((st_bs**2, 1))


                    contrast_loss += bce(neg, torch.zeros_like(neg)).mean()

                    contrast_acc = torch.gt(pos,0.0).float().mean()*0.5 + torch.lt(neg, 0.0).float().mean()*0.5

                    #contrast_loss += (pos_dist - neg_dist * torch.lt(neg_dist,10.0).float()).mean()
                    #contrast_acc = torch.lt(pos,5.0).float().mean()*0.5 + torch.gt(neg, 5.0).float().mean()*0.5



                else:
                    st_d, _, vql1 = enc.vq(F.normalize(enc.contrastive(zt_d)).unsqueeze(1))
                    st_pos, _, vql2 = enc.vq(F.normalize(enc.contrastive(enc(xt_pos).detach())).unsqueeze(1))
                    st_neg, _, vql3 = enc.vq(F.normalize(enc.contrastive(enc(xt_neg).detach())).unsqueeze(1))

                    st_d = st_d.squeeze(1)
                    st_pos = st_pos.squeeze(1)
                    st_neg = st_neg.squeeze(1)

                    contrast_loss += (vql1 + vql2 + vql3)

                    pos = torch.bmm(st_d.unsqueeze(1), st_pos.unsqueeze(2)).squeeze(-1)
                    neg = torch.bmm(st_d.unsqueeze(1), st_neg.unsqueeze(2)).squeeze(-1)

                    bce = nn.BCEWithLogitsLoss()

                    contrast_loss += (neg**2 - 2 * pos).mean()
                    contrast_acc = torch.gt(pos,0.5).float().mean()*0.5 + torch.lt(neg, 0.5).float().mean()*0.5


                inv_pred = enc.contrast_inv(st_d.detach())
                contrast_loss += F.mse_loss(inv_pred, zt_d)

                loss += contrast_loss

            if args.dist_learn:
                xt_d, _, xtk_d, k_d, _, _, _ = sample_batch(X, A, ast, est, 128, max_k=2000-1)
                st_d = enc(xt_d).detach()
                stk_d = enc(xtk_d).detach()

                dist_pred_loss = dist_pred.loss(st_d, stk_d, k_d)

                loss += dist_pred_loss


            loss += ac_loss + ap_loss + ep_loss + z_loss
            loss.backward()

            opt.step()
            opt.zero_grad()

            ema_forward.update()
            ema_a_probe.update()
            ema_enc.update()

            if True and (j % 5000 == 0 or j == 200000 - 1):
                print(j, 'AC_loss', ac_loss.item(), 'A_loss', ap_abserr.item(), 'Asqr_loss', ap_loss.item(), 'Z_loss',
                      z_loss.item())
                loss_log['AC_loss'].append(ac_loss.item())
                loss_log['A_loss'].append(ap_abserr.item())
                loss_log['Asqr_loss'].append(ap_loss.item())
                loss_log['Z_loss'].append(z_loss.item())

                if args.contrastive:
                    print('contrast-loss', contrast_loss, 'contrast-acc', contrast_acc)
                    loss_log['contrast_loss'].append(contrast_loss.item())
                    loss_log['contrast_acc'].append(contrast_acc.item())

                if args.use_wandb:
                    wandb.log(
                        {'update': j,
                         'ac-loss': ac_loss.item(),
                         'a-loss': ap_abserr.item(),
                         'asqr-loss': ap_loss.item()})

                    # print('forward test')
                    # print('true[t]', astate[0])
                    # print('s[t]', a_probe.enc(st)[0], 'a[t]', a[0])
                    # print('s[t+1]', a_probe.enc(stn)[0], 'z[t+1]', a_probe.enc(z_pred)[0])

            ema_a_probe.eval()
            ema_forward.eval()
            ema_enc.eval()


            def distplot():
                print("visualize distances!")

                xs = []
                xl = []
                for a in range(2, 99, 2):
                    for b in range(2, 99, 2):
                        start_true_s = [0.45, 0.1]
                        true_s = [a * 1.0 / 100, b * 1.0 / 100]
                        x = env.synth_obs(ap=true_s)
                        xs_obs = env.synth_obs(ap=start_true_s)
                        xl.append(x)
                        xs.append(xs_obs)

                xl = torch.Tensor(xl).to(device)
                xs = torch.Tensor(xs).to(device)

                zl = ema_enc(xl)
                zs = ema_enc(xs)

                dist_vals = dist_pred.predict_k(zl,zs)

                sl_probe = ema_a_probe(zl)
                ss_probe = ema_a_probe(zs)

                print('shapes', dist_vals.shape, sl_probe.shape, ss_probe.shape)

                plt.clf()
                plt.scatter(sl_probe.data.cpu()[:,0], sl_probe.data.cpu()[:,1], c=dist_vals.data.cpu())
                plt.colorbar()
                plt.legend()
                plt.savefig(join(field_folder, rf"distmap.jpg"))

            def vectorplot(a_use, name):

                fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
                fontdict = {'fontsize': 28, 'fontweight': 'bold'}

                # make grid
                action = []
                xl = []
                for a in range(2, 99, 5):
                    for b in range(2, 99, 10):
                        action.append(a_use)
                        true_s = [a * 1.0 / 100, b * 1.0 / 100]
                        x = env.synth_obs(ap=true_s)
                        xl.append(x)

                action = torch.Tensor(np.array(action)).to(device)
                xl = torch.Tensor(xl).to(device)
                print(xl.shape, action.shape)
                zt = ema_enc(xl)
                ztn = ema_forward(zt, action)
                st_inf = ema_a_probe(zt)
                stn_inf = ema_a_probe(ztn)
                print('st', st_inf[30], 'stn', stn_inf[30])

                px = st_inf[:, 0]
                py = st_inf[:, 1]
                pu = stn_inf[:, 0] - st_inf[:, 0]
                pv = stn_inf[:, 1] - st_inf[:, 1]

                # plot the quivers
                ax1.grid('on')
                ax1.plot(px.data.cpu(), py.data.cpu(), linewidth=1, color=next(colors))
                ax1.quiver(px.data.cpu(), py.data.cpu(), 0.5 * pu.data.cpu(), 0.5 * pv.data.cpu())
                ax1.set_title(name + " " + str(a_use))

                ax1.set_ylabel(rf"y (pixels)", fontdict=fontdict)
                ax1.set_xlabel(rf"x (pixels)", fontdict=fontdict)
                ax1.tick_params(axis='both', which='major', labelsize=28)
                ax1.tick_params(axis='both', which='minor', labelsize=18)
                ax1.set_title(rf"State Trajectories: {name} {a_use}.", fontdict=fontdict)
                ax1.legend(loc="center left", fontsize=8)

                fig.savefig(join(field_folder, rf"field_{name}.jpg"), dpi=79, bbox_inches='tight', facecolor='None')

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.clf()
                time.sleep(.01)

                return xl, action


            def squareplot(x_r, a_r):

                true_s = [0.4, 0.4]
                xl = env.synth_obs(ap=true_s)
                xl = torch.Tensor(xl).to(device).unsqueeze(0)

                xl = torch.cat([xl, x_r], dim=0)

                zt = ema_enc(xl)

                st_lst = []

                a_lst = [[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1], [-0.1, 0.0], [-0.1, 0.0], [0.0, -0.1],
                         [0.0, -0.1],
                         [0.0, 0.0], [0.0, 0.0]]
                for a in a_lst:
                    action = torch.Tensor(np.array(a)).to(device).unsqueeze(0)
                    action = torch.cat([action, a_r], dim=0)
                    st = ema_a_probe(zt)
                    st_lst.append(st.data.cpu()[0:1])
                    zt = ema_forward(zt, action)
                    print('st', st[0:1])
                    print('action', a)

                st_lst = torch.cat(st_lst, dim=0)

                true_sq = np.array(
                    [[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6], [0.4, 0.6], [0.4, 0.5],
                     [0.4, 0.4], [0.4, 0.4]])

                fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                fontdict = {'fontsize': 28, 'fontweight': 'bold'}

                ax.grid('on')

                ax.plot(st_lst[:, 0].numpy(), st_lst[:, 1].numpy(), linewidth=2, color=next(colors))
                ax.plot(true_sq[:, 0], true_sq[:, 1], linewidth=2, color="magenta")
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 1)
                ax.set_ylabel(rf"y (pixels)", fontdict=fontdict)
                ax.set_xlabel(rf"x (pixels)", fontdict=fontdict)
                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.tick_params(axis='both', which='minor', labelsize=18)

                ax.set_title("Square Plan", fontdict=fontdict)
                fig.savefig(join(plan_folder, f"plan.jpg"), dpi=79, bbox_inches='tight', facecolor='None')
                plt.clf()


            if True and j % 1000 == 0:
                vectorplot([0.0, 0.1], 'up')
                vectorplot([0.0, -0.1], 'down')
                vectorplot([-0.1, 0.0], 'left')
                vectorplot([0.1, 0.0], 'right')
                vectorplot([0.1, 0.1], 'up-right')
                x_r, a_r = vectorplot([-0.1, -0.1], 'down-left')

                squareplot(x_r, a_r)

                distplot()

                if args.use_wandb:
                    wandb.log({
                        'fields/down': wandb.Image(join(field_folder, "field_down.jpg")),
                        'fields/up': wandb.Image(join(field_folder, "field_up.jpg")),
                        'fields/left': wandb.Image(join(field_folder, "field_left.jpg")),
                        'fields/right': wandb.Image(join(field_folder, "field_right.jpg")),
                        'fields/up-right': wandb.Image(join(field_folder,
                                                            "field_up-right.jpg")),
                        'fields/plan': wandb.Image(join(plan_folder,
                                                        "plan.jpg")),
                        'update': j
                    })

                # save
                torch.save({'ac': ac.state_dict(),
                            'enc': enc.state_dict(),
                            'forward': forward.state_dict(),
                            'a_probe': a_probe.state_dict(),
                            'b_probe': b_probe.state_dict(),
                            'e_probe': e_probe.state_dict()}, model_path)
                
                torch.save(loss_log, train_data_path)

    elif args.opr == 'cluster-latent':

        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        a_probe.load_state_dict(model['a_probe'])
        enc = enc.eval().to(device)
        a_probe = a_probe.eval().to(device)

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)

        cluster_trans = Cluster_Transform(enc, forward, device=device)

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        print(f"Loading dataset from {dataset_path}")
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # manually check obstacle detection
        plt.figure()
        k = 0
        grounded_traj = ast[k * 100:(k + 30) * 100]

        # plot the obstacle
        if isinstance(env, RoomPolygonObstacleEnv):
            for obs in env.obs_lst:
                x_coords, y_coords = obs.exterior.xy
                plt.plot(x_coords, y_coords, color='k')

        plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color='blue', linewidth=0.3)
        plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], s=2, color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.savefig(os.path.join(logdir, 'polygon_obstacle_detection_check.png'), dpi=600, format='png')
        plt.clf()

        print('data loaded')

        # generate latent-states and ground them
        aug_latent_states = []
        predicted_grounded_states = []
        for i in tqdm(range(0, X.shape[0], 256)):
            with torch.no_grad():
                _aug_latent_state = cluster_trans.cluster_label_transform(X[i:i + 256], do_augment=args.use_augmented_latent_clustering)
                
                aug_latent_states += _aug_latent_state.cpu().numpy().tolist()
                predicted_grounded_states += a_probe(enc.contrast_inv(_aug_latent_state[:, :256])).cpu().numpy().tolist()

        predicted_grounded_states = np.array(predicted_grounded_states)
        grounded_states = np.array(ast[:len(aug_latent_states)])
        aug_latent_states = np.array(aug_latent_states)
        latent_state = aug_latent_states[:, :256]

        print('about to run kmeans')

        # clustering
        kmeans = KMeans(n_clusters=args.nclusters, random_state=0).fit(aug_latent_states)
        predicted_labels = kmeans.predict(aug_latent_states)
        centroids = a_probe(enc.contrast_inv(torch.FloatTensor(kmeans.cluster_centers_[:, :256]).to(device))).cpu().detach().numpy()

        # visualize and save
        kmean_plot_fig = plt.figure()
        plt.scatter(x=grounded_states[:, 0],
                    y=grounded_states[:, 1],
                    c=predicted_labels,
                    marker='.',
                    cmap='gist_ncar')
        plt.scatter(x=centroids[:, 0],
                    y=centroids[:, 1],
                    marker="*")

        for centroid_i, centroid in enumerate(centroids):
            plt.text(centroid[0], centroid[1], str(centroid_i), horizontalalignment='center', fontsize=8, color='black')

        pickle.dump(
            {'kmeans': kmeans, 'kmeans-plot': copy.deepcopy(kmean_plot_fig), 'grounded-cluster-center': centroids},
            open(os.path.join(logdir, 'kmeans_info.p'), 'wb'))
        plt.savefig(join(field_folder, 'latent_cluster.png'))
        plt.clf()

        np.savez_compressed(os.path.join(data_folder, 'latent.npz'), aug_latent_states=aug_latent_states)
        np.savez_compressed(os.path.join(data_folder, 'grounded_states.npz'), grounded_states=grounded_states)
        np.savez_compressed(os.path.join(data_folder, 'predicted_labels.npz'), predicted_labels=predicted_labels)

        #plt.scatter(x=grounded_states[:, 0],
        #            y=predicted_grounded_states[:, 0],
        #            marker='.')
        #plt.savefig(join(field_folder, 'ground_vs_predicted_state.png'))
        if args.use_wandb:
            wandb.log({'latent-cluster': wandb.Image(join(field_folder, "latent_cluster.png")),
                       'grounded-vs-predicted-state': wandb.Image(join(field_folder, "ground_vs_predicted_state.png"))})
            wandb.save(glob_str='kmeans.p', policy='now')
    elif args.opr == 'generate-mdp':
        # load model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval()

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)
        cluster_trans = Cluster_Transform(enc, forward, device=device)

        # load clustering
        kmeans_info = pickle.load(open(os.path.join(logdir, 'kmeans_info.p'), 'rb'))
        kmeans = kmeans_info['kmeans']
        kmeans_fig = kmeans_info['kmeans-plot']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # generate latent-states and find corresponding label
        latent_states, states_label = [], []
        for i in tqdm(range(0, len(X), 256)):
            with torch.no_grad():
                _aug_latent_state = cluster_trans.cluster_label_transform(X[i:i + 256], do_augment=args.use_augmented_latent_clustering)
                #if args.use_augmented_latent_clustering:
                #    _aug_latent_state = cluster_trans.cluster_label_transform(X[i:i + 256])
                #else:
                #    _aug_latent_state = enc(torch.FloatTensor(X[i:i + 256]).to(device))
                latent_states += _aug_latent_state[:, :256].cpu().numpy().tolist()
                states_label += kmeans.predict(_aug_latent_state.cpu().numpy().tolist()).tolist()

        next_state = np.array(states_label[1:])
        next_state = next_state[np.abs(A[:-1]).sum(1) < 0.1]
        states_label = np.array(states_label[:-1])[np.abs(A[:-1]).sum(1) < 0.1]
        A = A[:-1]
        A = A[np.abs(A).sum(1) < 0.1]

        print(states_label.shape, A.shape, next_state.shape)

        empirical_mdp = EmpiricalMDP(state=states_label,
                                     action=A,
                                     next_state=next_state,
                                     reward=np.zeros_like(A))

        # draw action vectors on cluster-mdp
        for cluster_i, cluster_center in tqdm(enumerate(grounded_cluster_centers), total=len(grounded_cluster_centers)):

            print('cluster', cluster_i)

            if not cluster_i in empirical_mdp.unique_states_dict:
                continue

            for action in [_ for _ in empirical_mdp.transition[empirical_mdp.unique_states_dict[cluster_i]] if
                           not np.isnan(_).all()]:
                plt.quiver(cluster_center[0], cluster_center[1], action[0], action[1])
                print('quiver', cluster_center[0], cluster_center[1], action[0], action[1])
        plt.savefig(join(field_folder, 'latent_cluster_with_action_vector.png'))
        plt.clf()

        transition_img = empirical_mdp.visualize_transition(save_path=join(field_folder, 'transition_img.png'))
        # save
        pickle.dump(empirical_mdp, open(os.path.join(logdir, 'empirical_mdp.p'), 'wb'))

        # save
        if args.use_wandb:
            wandb.log({'mdp': wandb.Image(join(field_folder, "transition_img.png"))})
            # wandb.log({'latent-cluster-with-action-vector': wandb.Image(join(field_folder,'latent_cluster_with_action_vector.png'))})
            wandb.save(glob_str='empirical_mdp.p', policy="now")

    elif args.opr == 'high-low-plan':
        # load abstract mdp
        mdp_path = os.path.join(logdir, 'empirical_mdp.p')
        empirical_mdp = pickle.load(open(mdp_path, 'rb'))

        # load models
        model_path = os.path.join(data_folder, 'model.p')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval().to(device)

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)

        # latent space dynamics
        dynamics = LatentWrapper(forward)
        dynamics.eval()

        cluster_trans = Cluster_Transform(enc, forward, device=device)

        a_probe.load_state_dict(model['a_probe'])

        # load clustering
        kmeans_info = pickle.load(open(os.path.join(logdir, 'kmeans_info.p'), 'rb'))
        kmeans = kmeans_info['kmeans']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # load-dataset
        dataset_path = os.path.join(data_folder, 'dataset.p')
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # # SC: need to double check
        X = X[:-1]
        ast = ast[:-1]
        A = A[:-1]

        X = X[np.abs(A).sum(1) < 0.1]
        ast = ast[np.abs(A).sum(1) < 0.1]
        A = A[np.abs(A).sum(1) < 0.1]

        # initialization
        exp_id = args.exp_id
        traj_opt_fig_dir = os.path.join(logdir, 'high_low_data')
        os.makedirs(traj_opt_fig_dir, exist_ok=True)

        # specify start state and goal state
        from_to = args.from_to
        if not isinstance(from_to, list):
            from_to = [13, 2]

        # initial mdp state
        init_mdp_state = from_to[0]
        init_obs, init_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=init_mdp_state)
        init_lat_state = enc(torch.FloatTensor(init_obs).to(device).unsqueeze(0))

        nz = init_lat_state.size(1)
        nu = ast.shape[1]

        target_mdp_state = from_to[1]
        target_obs, target_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, abstract_state=target_mdp_state)
        target_lat_state = enc(torch.FloatTensor(target_obs).to(device).unsqueeze(0))

        # high low level plan 
        from traj_opt.room_planner import room_high_low_planner, room_low_planner
        traj_opt_data_path = os.path.join(logdir, 'high_low_data', f'high_low_{exp_id}.p')
        high_low_mpc_data = room_high_low_planner(env, nz, nu, enc, dynamics, a_probe, empirical_mdp, kmeans, cluster_trans,
                          init_gt_agent_state, init_lat_state, init_mdp_state, 
                          target_gt_agent_state, target_lat_state, target_mdp_state,
                          save_path = traj_opt_data_path, augmented = args.use_augmented_latent_clustering)
        
        traj_opt_data_path = os.path.join(logdir, 'high_low_data', f'low_{exp_id}.p')
        low_mpc_data = room_low_planner(env, nz, nu, enc, dynamics, a_probe, empirical_mdp, kmeans, cluster_trans,
                          init_gt_agent_state, init_lat_state, init_mdp_state, 
                          target_gt_agent_state, target_lat_state, target_mdp_state,
                          save_path = traj_opt_data_path, augmented = args.use_augmented_latent_clustering)
        
        high_low_runtime = sum(high_low_mpc_data['mpc_time'])/len(high_low_mpc_data['mpc_time'])
        print(f'avg. runtime of high-low planner: {high_low_runtime} s')

        low_runtime = sum(low_mpc_data['mpc_time'])/len(low_mpc_data['mpc_time'])
        print(f'avg. runtime of low-level planner: {low_runtime} s')

        plt.figure()
       
        for obstacle in env.obs_lst:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, 'k-')


        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        # plt.grid()

        plt.savefig(os.path.join(logdir, 'env.png'), dpi = 600)
        plt.clf()
       
        # plot the trajectory
        fig, ax = plt.subplots()
        grounded_traj = high_low_mpc_data['grounded_states']
        target_grounded_state_np = target_gt_agent_state
        action_log = high_low_mpc_data['actions'].detach().cpu().numpy()

        grounded_traj_low = low_mpc_data['grounded_states']

        for obstacle in env.obs_lst:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, 'k-')

        plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color='blue', linewidth=3.0, label='high+low')
        plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], color='blue')

        plt.plot(grounded_traj_low[:, 0], grounded_traj_low[:, 1], '-.', linewidth=3.0, color='green', label='low')
        plt.scatter(grounded_traj_low[:, 0], grounded_traj_low[:, 1], color='green')

        plt.scatter(grounded_traj[0, 0], grounded_traj[0, 1], marker='o', s=50, color='k', label='init')
        plt.scatter(target_grounded_state_np[0], target_grounded_state_np[1], marker='s',s=50, color='r', label='target')
        
        plt.legend(loc='upper right', ncol=2, fancybox=True, shadow=False, fontsize=18)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        # plt.grid()

        ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        fig.tight_layout()

        # xlabels = ax.get_xticks().tolist()
        # xlabels = [item.get_text() for item in ax.get_xticklabels()]
        # xlabels[0] = 'change'
        # ax.set_xticklabels(xlabels)
        plt.text(-0.2,-0.2, '0.0', fontsize = 16)
        
        fig_path = os.path.join(traj_opt_fig_dir, f'{exp_id}.png')
        print(f"Saved figure to {fig_path}")
        plt.savefig(os.path.join(traj_opt_fig_dir, f'{exp_id}.png'), dpi = 600)
        plt.clf()

    elif args.opr == 'evaluate-planners':
        task_dir = 'tasks'
        eval_dir = os.path.join(task_dir, args.env, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        test_set_path = os.path.join(eval_dir, f'test_set_{args.env}.p')
        
        print(f"Loading test set from: {test_set_path}")
        test_set_data = pickle.load(open(test_set_path, 'rb'))
        test_cases = test_set_data['test_cases']

        mdp_path = os.path.join(logdir, 'empirical_mdp.p')
        empirical_mdp = pickle.load(open(mdp_path, 'rb'))

        model_path = os.path.join(data_folder, 'model.p') 
        model = torch.load(model_path, map_location=torch.device('cpu'))
        enc.load_state_dict(model['enc'])
        enc.eval().to(device)

        forward.load_state_dict(model['forward'])
        forward.eval().to(device)

        dynamics = LatentWrapper(forward)
        dynamics.eval()

        cluster_trans = Cluster_Transform(enc, forward, device=device)
        a_probe.load_state_dict(model['a_probe'])

        kmeans_path = os.path.join(logdir, 'kmeans_info.p')
        kmeans_info = pickle.load(open(kmeans_path, 'rb'))
        kmeans = kmeans_info['kmeans']
        grounded_cluster_centers = kmeans_info['grounded-cluster-center']

        # success threshold
        SUCCESS_THRESHOLD = 0.030
        
        # Evaluation results storage
        results = {
            'high_low': {
                'successes': 0,
                'total_time': [],
                'path_lengths': [],
                'final_distances': [],
                'episodes_data': []
            },
            'low': {
                'successes': 0,
                'total_time': [],
                'path_lengths': [],
                'final_distances': [],
                'episodes_data': []
            },
            'test_set_info': test_set_data,
            'evaluation_seed': args.seed,
            'exp_id': args.exp_id,
            'timestamp': time.time()
        }
        
        from traj_opt.room_planner import room_high_low_planner, room_low_planner

        for test_case in tqdm(test_cases, desc=f"Evaluating {args.env} planners"):
            episode_idx = test_case['test_id']
            start_pos = test_case['start_pos']
            goal_pos = test_case['goal_pos']
            
            print(f"\n=== Test {episode_idx + 1}/{len(test_cases)} ===")
            print(f"Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}], Goal: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}]")
            
            start_obs = env.synth_obs(start_pos)
            goal_obs = env.synth_obs(goal_pos) 
            
            start_lat_state = enc(torch.FloatTensor(start_obs).to(device))
            goal_lat_state = enc(torch.FloatTensor(goal_obs).to(device))
            
            start_aug_latent = cluster_trans.cluster_label_transform_latent(start_lat_state, do_augment=args.use_augmented_latent_clustering)
            goal_aug_latent = cluster_trans.cluster_label_transform_latent(goal_lat_state, do_augment=args.use_augmented_latent_clustering)
            
            start_mdp_state = kmeans.predict(start_aug_latent.detach().cpu())[0]
            goal_mdp_state = kmeans.predict(goal_aug_latent.detach().cpu())[0]
            
            print(f"Abstract states - Start: {start_mdp_state}, Goal: {goal_mdp_state}")
            
            planners = [
                ('high_low', room_high_low_planner),
                ('low', room_low_planner)
            ]
            
            for planner_name, planner_func in planners:
                print(f"\n--- Testing {planner_name} planner ---")
                
                try:
                    # Run planner
                    planner_start_time = time.time()
                    mpc_data = planner_func(
                        env, start_lat_state.shape[1], 2, enc, dynamics, a_probe, 
                        empirical_mdp, kmeans, cluster_trans,
                        np.array(start_pos), start_lat_state, start_mdp_state,
                        np.array(goal_pos), goal_lat_state, goal_mdp_state,
                        save_path=None, augmented=args.use_augmented_latent_clustering
                    )
                    total_time = time.time() - planner_start_time
                    
                    # Extract results
                    final_pos = mpc_data['grounded_states'][-1]
                    final_distance = np.linalg.norm(final_pos - np.array(goal_pos))
                    path_length = np.sum([np.linalg.norm(mpc_data['grounded_states'][i+1] - mpc_data['grounded_states'][i]) 
                                        for i in range(len(mpc_data['grounded_states'])-1)])
                    
                    # Check success
                    success = final_distance <= SUCCESS_THRESHOLD
                    num_steps = len(mpc_data['grounded_states']) - 1
                    
                    # Store results
                    results[planner_name]['final_distances'].append(final_distance)
                    results[planner_name]['total_time'].append(total_time)
                    results[planner_name]['path_lengths'].append(path_length)
                    results[planner_name]['episodes_data'].append({
                        'test_id': episode_idx,
                        'start_pos': start_pos,
                        'goal_pos': goal_pos,
                        'final_pos': final_pos.tolist(),
                        'success': success,
                        'final_distance': final_distance,
                        'path_length': path_length,
                        'total_time': total_time,
                        'num_steps': num_steps,
                        'trajectory': mpc_data['grounded_states'].tolist(),
                        'start_mdp_state': start_mdp_state,
                        'goal_mdp_state': goal_mdp_state
                    })
                    
                    if success:
                        results[planner_name]['successes'] += 1
                    
                    print(f"{planner_name}: Success={success}, Final dist={final_distance:.4f}, Steps={num_steps}, Time={total_time:.2f}s")
                    
                except Exception as e:
                    print(f"ERROR with {planner_name} planner: {e}")
                    import traceback
                    traceback.print_exc()
                    # Record failure
                    results[planner_name]['episodes_data'].append({
                        'test_id': episode_idx,
                        'start_pos': start_pos,
                        'goal_pos': goal_pos,
                        'success': False,
                        'error': str(e),
                        'start_mdp_state': start_mdp_state,
                        'goal_mdp_state': goal_mdp_state
                    })

        # Calculate and print final results
        NUM_EVAL_EPISODES = len(test_cases)
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS")
        print(f"Environment: {args.env} | Seed: {args.seed} | Exp ID: {args.exp_id}")
        print("="*80)
        
        summary_stats = {}
        
        for planner_name in ['high_low', 'low']:
            data = results[planner_name]
            success_rate = data['successes'] / NUM_EVAL_EPISODES * 100
            
            print(f"\n{planner_name.upper()} PLANNER:")
            print(f"  Success Rate: {success_rate:.1f}% ({data['successes']}/{NUM_EVAL_EPISODES})")
            
            summary_stats[f'{planner_name}_success_rate'] = success_rate
            
            if len(data['final_distances']) > 0:
                avg_final_dist = np.mean(data['final_distances'])
                avg_path_length = np.mean(data['path_lengths'])
                avg_runtime = np.mean(data['total_time'])
                
                print(f"  Average Final Distance: {avg_final_dist:.4f} ± {np.std(data['final_distances']):.4f}")
                print(f"  Average Path Length: {avg_path_length:.4f} ± {np.std(data['path_lengths']):.4f}")
                print(f"  Average Runtime: {avg_runtime:.2f}s ± {np.std(data['total_time']):.2f}s")
                
                summary_stats[f'{planner_name}_avg_final_dist'] = avg_final_dist
                summary_stats[f'{planner_name}_avg_path_length'] = avg_path_length
                summary_stats[f'{planner_name}_avg_runtime'] = avg_runtime
            
            # Success-only statistics
            success_episodes = [ep for ep in data['episodes_data'] if ep.get('success', False)]
            if success_episodes:
                success_times = [ep['total_time'] for ep in success_episodes]
                success_distances = [ep['final_distance'] for ep in success_episodes]
                success_steps = [ep['num_steps'] for ep in success_episodes if 'num_steps' in ep]
                
                avg_success_time = np.mean(success_times)
                avg_success_dist = np.mean(success_distances)
                
                print(f"  Successful Episodes Only:")
                print(f"    - Avg Time: {avg_success_time:.2f}s ± {np.std(success_times):.2f}s")
                print(f"    - Avg Final Dist: {avg_success_dist:.4f} ± {np.std(success_distances):.4f}")
                if success_steps:
                    avg_success_steps = np.mean(success_steps)
                    print(f"    - Avg Steps: {avg_success_steps:.1f} ± {np.std(success_steps):.1f}")
                    summary_stats[f'{planner_name}_avg_success_steps'] = avg_success_steps
                
                summary_stats[f'{planner_name}_avg_success_time'] = avg_success_time
                summary_stats[f'{planner_name}_avg_success_dist'] = avg_success_dist

        # Add summary stats to results
        results['summary_stats'] = summary_stats

        # Save detailed results
        results_filename = f'evaluation_results_{args.env}_seed{args.seed}_{args.exp_id}.p'
        results_path = os.path.join(eval_dir, results_filename)
        pickle.dump(results, open(results_path, 'wb'))
        print(f"\nDetailed results saved to: {results_path}")

        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rates
        planners = ['high_low', 'low']
        success_rates = [results[p]['successes']/NUM_EVAL_EPISODES*100 for p in planners]
        bars = axes[0,0].bar(planners, success_rates, color=['r', 'b'], alpha=0.8)
        axes[0,0].set_ylabel('Success Rate (%)', fontsize=12)
        axes[0,0].set_title(f'Success Rates - Seed {args.seed}', fontsize=14)
        axes[0,0].set_ylim(0, 100)
        # Add percentage labels on bars
        for bar, rate in zip(bars, success_rates):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
        
        # Final distances
        colors = ['r','b']
        for i, planner in enumerate(planners):
            if len(results[planner]['final_distances']) > 0:
                axes[0,1].hist(results[planner]['final_distances'], alpha=0.6, 
                            label=planner.replace('_', '-'), bins=20, color=colors[i])
        axes[0,1].axvline(SUCCESS_THRESHOLD, color='red', linestyle='--', linewidth=2, label='Success Threshold')
        axes[0,1].set_xlabel('Final Distance to Goal', fontsize=12)
        axes[0,1].set_ylabel('Count', fontsize=12)
        axes[0,1].set_title('Distribution of Final Distances', fontsize=14)
        axes[0,1].legend()
        
        # Runtime comparison
        runtime_data = [results[p]['total_time'] for p in planners if len(results[p]['total_time']) > 0]
        labels = [p.replace('_', '-') for p in planners]
        if len(runtime_data) == 2:
            bp = axes[1,0].boxplot(runtime_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1,0].set_ylabel('Runtime (seconds)', fontsize=12)
            axes[1,0].set_title('Runtime Comparison', fontsize=14)
        
        # Path lengths
        pathlength_data = [results[p]['path_lengths'] for p in planners if len(results[p]['path_lengths']) > 0]
        if len(pathlength_data) == 2:
            bp = axes[1,1].boxplot(pathlength_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1,1].set_ylabel('Path Length', fontsize=12)
            axes[1,1].set_title('Path Length Comparison', fontsize=14)
        
        plt.suptitle(f'Evaluation Summary - {args.env} - Seed {args.seed} - {args.exp_id}', fontsize=16)
        plt.tight_layout()
        
        plot_filename = f'evaluation_summary_{args.env}_seed{args.seed}_{args.exp_id}.png'
        plot_path = os.path.join(eval_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to: {plot_path}")
        
        if args.use_wandb:
            wandb.log({
                f'eval_seed{args.seed}/high_low_success_rate': summary_stats['high_low_success_rate'],
                f'eval_seed{args.seed}/low_success_rate': summary_stats['low_success_rate'],
                f'eval_seed{args.seed}/summary_plot': wandb.Image(plot_path)
            })
        
        print(f"\nEvaluation complete!")

    else:
        raise ValueError()
