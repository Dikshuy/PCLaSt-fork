import os
import time
import random
import argparse
import pickle
from datetime import datetime
from os.path import join
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import wandb
from ema_pytorch import EMA

# Environment imports
from room_env import RoomEnv
from room_obstacle_env import RoomObstacleEnv
from room_polygon_obstacle_env import (
    RoomPolygonObstacleEnv, RoomMultiPassageEnv, 
    RoomSpiral, RoomMultiPassageEnvLarge
)
from models import Encoder, Probe, AC, LatentForward
from dist_pred_model import DistPred
from emprical_mdp import EmpiricalMDP
from dijkstra import make_ls, DP_goals


class LatentWrapper(nn.Module):
    """Wrapper for latent forward dynamics"""
    def __init__(self, latent):
        super().__init__()
        self.latent = latent
        self.nu = 2

    def forward(self, z, a):
        return self.latent(z, a, detach=False)


class ClusterTransform:
    """Optimized clustering transformation handler"""
    def __init__(self, enc, fwd_dyn, device='cuda'):
        self.enc = enc
        self.fwd_dyn = fwd_dyn
        self.device = device
        # Pre-compute actions
        self.actions = torch.FloatTensor([
            [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]
        ]).to(device)

    def cluster_label_transform(self, X, do_augment=False, contrastive=True):
        """Transform observations to latent states for clustering"""
        with torch.no_grad():
            # Ensure X is a proper numpy array
            if isinstance(X, list):
                X = np.array(X, dtype=np.float32)
            elif not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)
            
            # Convert to tensor efficiently
            img = torch.from_numpy(X).to(self.device)
            z = self.enc(img)
            
            output = self.enc.contrastive(z) if contrastive else z
            
            if do_augment:
                # Vectorized augmentation
                batch_size = z.size(0)
                augmented = []
                for action in self.actions:
                    action_batch = action.unsqueeze(0).expand(batch_size, -1)
                    y = self.fwd_dyn(z, action_batch)
                    if contrastive:
                        y = self.enc.contrastive(y)
                    augmented.append(y)
                output = torch.cat([output] + augmented, dim=1)
        
        return output


def sample_batch(X, A, ast, est, bs, max_k, device):
    """Optimized batch sampling with vectorized operations"""
    N = X.shape[0]
    # Vectorized sampling
    indices = np.random.randint(0, N - max_k - 1, bs)
    k_values = np.random.randint(1, max_k + 1, bs)
    
    # Direct indexing with numpy arrays
    xt = torch.from_numpy(X[indices].astype(np.float32)).to(device)
    xtn = torch.from_numpy(X[indices + 1].astype(np.float32)).to(device)
    xtk = torch.from_numpy(X[indices + k_values].astype(np.float32)).to(device)
    klst = torch.from_numpy(k_values.astype(np.int64)).to(device)
    alst = torch.from_numpy(A[indices].astype(np.float32)).to(device)
    astate = torch.from_numpy(ast[indices].astype(np.float32)).to(device)
    estate = torch.from_numpy(est[indices].astype(np.float32)).to(device)
    
    return xt, xtn, xtk, klst, alst, astate, estate


def obs_sampler(dataset_obs, dataset_agent_states, state_labels, abstract_state):
    """Optimized observation sampler"""
    mask = state_labels == abstract_state
    filtered_obs = dataset_obs[mask]
    filtered_agent_states = dataset_agent_states[mask]
    
    if len(filtered_obs) == 0:
        raise ValueError(f"No observations found for state {abstract_state}")
    
    index = np.random.randint(len(filtered_obs))
    return filtered_obs[index], filtered_agent_states[index]


def create_environment(env_name):
    """Environment factory"""
    env_map = {
        'room': RoomEnv,
        'obstacle': RoomObstacleEnv,
        'polygon-obs': RoomPolygonObstacleEnv,
        'room-multi-passage': RoomMultiPassageEnv,
        'room-spiral': RoomSpiral,
        'room-multi-passage-large': RoomMultiPassageEnvLarge
    }
    
    if env_name == 'rat':
        from rat_env import RatEnvWrapper
        return RatEnvWrapper()
    
    return env_map[env_name]()


def setup_directories(args):
    """Setup and create necessary directories"""
    logdir = os.path.expanduser(args.logdir)
    logdir = join(logdir, args.env)
    
    paths = {
        'logdir': logdir,
        'field_folder': join(logdir, "fields"),
        'plan_folder': join(logdir, "fields"),
        'data_folder': join(logdir, 'data'),
        'train_data_folder': join(logdir, 'train_data')
    }
    
    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Add file paths
    paths['dataset_path'] = join(paths['data_folder'], 'dataset.p')
    paths['model_path'] = join(paths['data_folder'], 'model.p')
    paths['train_data_path'] = join(paths['train_data_folder'], 'loss_log.p')
    
    return paths


def main():
    parser = argparse.ArgumentParser()
    
    # Wandb setup
    wandb_args = parser.add_argument_group('wandb setup')
    wandb_args.add_argument('--wandb-project-name', default='acsnc')
    wandb_args.add_argument('--use-wandb', action='store_true')
    
    # Training arguments
    train_args = parser.add_argument_group('training setup')
    train_args.add_argument("--opr", default="high-low-plan",
                            choices=['generate-data', 'train', 'cluster-latent',
                                     'generate-mdp', 'high-low-plan'])
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
    train_args.add_argument("--env", default='polygon-obs')
    train_args.add_argument('--logdir', default='logs', type=str)
    train_args.add_argument('--exp_id', default='test', type=str)
    train_args.add_argument('--from_to', default=0, nargs="+", type=int)
    train_args.add_argument('--scaling_factor', default=1.0, type=float)
    train_args.add_argument("--seed", default=0, type=int)
    train_args.add_argument('--use-augmented-latent-clustering', action='store_true')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, save_code=True)
        wandb.config.update({x.dest: vars(args)[x.dest] 
                            for x in train_args._group_actions})
    
    # Create environment
    env = create_environment(args.env)
    
    # Initialize models
    models = {
        'ac': AC(din=args.latent_dim, nk=args.k_embedding_dim, nact=2).to(device),
        'enc': Encoder(100 * 100, args.latent_dim, args.ndiscrete).to(device),
        'forward': LatentForward(args.latent_dim, 2).to(device),
        'a_probe': Probe(args.latent_dim, 2).to(device),
        'b_probe': Probe(args.latent_dim, 2).to(device),
        'e_probe': Probe(args.latent_dim, 2).to(device),
        'dist_pred': DistPred(args.latent_dim, 2000).to(device)
    }
    
    # Initialize EMA models
    ema_models = {
        'enc': EMA(models['enc'], beta=0.99),
        'forward': EMA(models['forward'], beta=0.99),
        'a_probe': EMA(models['a_probe'], beta=0.99)
    }
    
    # Setup directories
    paths = setup_directories(args)
    
    # Execute operation
    if args.opr == 'generate-data':
        generate_data(env, args, paths)
    elif args.opr == 'train':
        train(env, models, ema_models, args, paths, device)
    elif args.opr == 'cluster-latent':
        cluster_latent(env, models, args, paths, device)
    elif args.opr == 'generate-mdp':
        generate_mdp(models, args, paths, device)
    elif args.opr == 'high-low-plan':
        high_low_plan(env, models, args, paths, device)


def generate_data(env, args, paths):
    """Generate training data"""
    print(f"Generating {args.num_data_samples} samples...")
    
    # Pre-allocate arrays for efficiency
    X = np.zeros((args.num_data_samples, 100, 100), dtype=np.float32)
    A = np.zeros((args.num_data_samples, 2), dtype=np.float32)
    ast = np.zeros((args.num_data_samples, 2), dtype=np.float32)
    est = np.zeros((args.num_data_samples, 2), dtype=np.float32)
    
    for i in tqdm(range(args.num_data_samples)):
        a = env.random_action()
        x, agent_state, exo_state = env.get_obs()
        env.step(a)
        
        X[i] = x
        A[i] = a
        ast[i] = agent_state
        est[i] = exo_state
    
    # Save data
    dataset = {'X': X, 'A': A, 'ast': ast, 'est': est}
    with open(paths['dataset_path'], 'wb') as f:
        pickle.dump(dataset, f)
    
    # Save individual components for faster loading
    np.save(join(paths['data_folder'], "images.npy"), X)
    np.save(join(paths['data_folder'], "actions.npy"), A)
    np.save(join(paths['data_folder'], "agent_state.npy"), ast)
    np.save(join(paths['data_folder'], "exo_state.npy"), est)
    
    print(f'Data generated and stored in {paths["dataset_path"]}')


def train(env, models, ema_models, args, paths, device):
    """Optimized training loop"""
    # Load dataset
    with open(paths['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loading dataset from {paths['dataset_path']}")
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
    
    # Initialize optimizer with all model parameters
    all_params = []
    for model in models.values():
        all_params.extend(model.parameters())
    opt = torch.optim.Adam(all_params, lr=0.0001)
    
    print(f'Num samples: {X.shape[0]}')
    
    # K-means clustering on actions
    print('Running K-means...')
    kmeans = KMeans(n_clusters=20, verbose=1, n_init=10).fit(A)
    A = np.concatenate([A, kmeans.labels_.reshape(-1, 1)], axis=1)
    
    # Initialize loss tracking
    loss_log = defaultdict(list)
    
    # Training loop
    for j in tqdm(range(200000)):
        # Set training mode
        for model in models.values():
            model.train()
        
        # Sample batch
        batch = sample_batch(X, A, ast, est, args.batch_size, args.max_k, device)
        xt, xtn, xtk, k, a, astate, estate = batch
        astate = torch.round(astate, decimals=3)
        
        # Forward pass - compute encodings once
        st = models['enc'](xt)
        stk = models['enc'](xtk)
        stn = models['enc'](xtn)
        
        # Compute losses
        ac_loss = models['ac'](st, stk, k, a)
        ap_loss, ap_abserr = models['a_probe'].loss(st, astate)
        z_loss, z_pred = models['forward'].loss(st, stn, a)
        
        total_loss = ac_loss + ap_loss + z_loss
        
        # Optional: Mixup regularization
        if args.do_mixup:
            mix_lamb = np.random.beta(0.5, 0.5)
            mix_ind = torch.randperm(st.shape[0])
            
            st_mix = st * mix_lamb + st[mix_ind] * (1 - mix_lamb)
            stn_mix = stn * mix_lamb + stn[mix_ind] * (1 - mix_lamb)
            z_loss_mix, _ = models['forward'].loss(st_mix, stn_mix, a, do_detach=False)
            
            total_loss += z_loss_mix
        
        # Optional: Contrastive loss
        if args.contrastive:
            contrast_loss, contrast_acc = compute_contrastive_loss(
                models, X, A, ast, est, args, device
            )
            total_loss += contrast_loss
            
            # Store for logging
            contrast_loss_val = contrast_loss.item()
            contrast_acc_val = contrast_acc.item()
        
        # Optional: Distance learning
        if args.dist_learn:
            dist_batch = sample_batch(X, A, ast, est, 128, 2000-1, device)
            xt_d, _, xtk_d, k_d, _, _, _ = dist_batch
            
            with torch.no_grad():
                st_d = models['enc'](xt_d)
                stk_d = models['enc'](xtk_d)
            
            dist_pred_loss = models['dist_pred'].loss(st_d, stk_d, k_d)
            total_loss += dist_pred_loss
        
        # Backward pass
        total_loss.backward()
        opt.step()
        opt.zero_grad()
        
        # Update EMA models
        for ema in ema_models.values():
            ema.update()
        
        # Logging and visualization
        if j % 5000 == 0 or j == 199999:
            log_training_progress(j, ac_loss, ap_loss, ap_abserr, z_loss, 
                                loss_log, args, paths)
            
            if args.contrastive:
                loss_log['contrast_loss'].append(contrast_loss_val)
                loss_log['contrast_acc'].append(contrast_acc_val)
                print(f'contrast-loss: {contrast_loss_val:.4f}, contrast-acc: {contrast_acc_val:.4f}')
        
        if j % 10000 == 0:
            visualize_training(env, models, ema_models, args, paths, device)
            save_checkpoint(models, paths['model_path'])
            
            with open(paths['train_data_path'], 'wb') as f:
                torch.save(loss_log, f)


def compute_contrastive_loss(models, X, A, ast, est, args, device):
    """Compute contrastive learning loss"""
    # Sample positive and negative examples
    batch_pos = sample_batch(X, A, ast, est, 128, args.contrastive_k, device)
    xt_d, _, xt_pos, _, _, _, _ = batch_pos
    
    batch_neg = sample_batch(X, A, ast, est, 128, 1, device)
    xt_neg = batch_neg[0]
    
    # Compute embeddings
    with torch.no_grad():
        zt_d = models['enc'](xt_d)
        zt_pos = models['enc'](xt_pos)
        zt_neg = models['enc'](xt_neg)
    
    # Contrastive projections
    st_d = models['enc'].contrastive(zt_d)
    st_pos = models['enc'].contrastive(zt_pos)
    st_neg = models['enc'].contrastive(zt_neg)
    
    # Optional: Vector quantization
    vq_loss = 0
    if hasattr(models['enc'], 'vq'):
        st_d, _, vql1 = models['enc'].vq(st_d.unsqueeze(1))
        st_pos, _, vql2 = models['enc'].vq(st_pos.unsqueeze(1))
        st_neg, _, vql3 = models['enc'].vq(st_neg.unsqueeze(1))
        st_d = st_d.squeeze(1)
        st_pos = st_pos.squeeze(1)
        st_neg = st_neg.squeeze(1)
        vq_loss = (vql1 + vql2 + vql3).mean()
    
    # Compute distances
    pos_dist = torch.sqrt(((st_d - st_pos)**2).sum(dim=1, keepdim=True))
    neg_dist = torch.sqrt(((st_d - st_neg)**2).sum(dim=1, keepdim=True))
    
    # Compute logits
    w = models['enc'].w_contrast.weight
    b = models['enc'].b_contrast.bias
    pos_logits = torch.exp(b) - torch.exp(w) * pos_dist
    neg_logits = torch.exp(b) - torch.exp(w) * neg_dist
    
    # Binary cross-entropy loss
    bce = nn.BCEWithLogitsLoss()
    pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
    neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
    
    # Inverse prediction loss
    inv_pred = models['enc'].contrast_inv(st_d.detach())
    inv_loss = F.mse_loss(inv_pred, zt_d)
    
    # Total loss and accuracy
    contrast_loss = pos_loss + neg_loss + inv_loss + vq_loss
    contrast_acc = (torch.gt(pos_logits, 0).float().mean() + 
                   torch.lt(neg_logits, 0).float().mean()) * 0.5
    
    return contrast_loss, contrast_acc


def log_training_progress(j, ac_loss, ap_loss, ap_abserr, z_loss, 
                         loss_log, args, paths):
    """Log training progress"""
    print(f"{j} AC_loss: {ac_loss.item():.4f} A_loss: {ap_abserr.item():.4f} "
          f"Asqr_loss: {ap_loss.item():.4f} Z_loss: {z_loss.item():.4f}")
    
    loss_log['AC_loss'].append(ac_loss.item())
    loss_log['A_loss'].append(ap_abserr.item())
    loss_log['Asqr_loss'].append(ap_loss.item())
    loss_log['Z_loss'].append(z_loss.item())
    
    if args.use_wandb:
        wandb.log({
            'update': j,
            'ac-loss': ac_loss.item(),
            'a-loss': ap_abserr.item(),
            'asqr-loss': ap_loss.item(),
            'z-loss': z_loss.item()
        })


def visualize_training(env, models, ema_models, args, paths, device):
    """Create visualization plots during training"""
    # Set eval mode for EMA models
    for ema in ema_models.values():
        ema.eval()
    
    # Vector field plots
    actions = {
        'up': [0.0, 0.1],
        'down': [0.0, -0.1],
        'left': [-0.1, 0.0],
        'right': [0.1, 0.0],
        'up-right': [0.1, 0.1],
        'down-left': [-0.1, -0.1]
    }
    
    for name, action in actions.items():
        plot_vector_field(env, ema_models['enc'].ema_model, 
                         ema_models['forward'].ema_model,
                         ema_models['a_probe'], action, name, device, paths)
    
    # Square trajectory plot
    plot_square_trajectory(env, ema_models, device, paths)
    
    # Distance visualization
    if args.dist_learn:
        plot_distance_map(env, ema_models, models['dist_pred'], device, paths)
    
    if args.use_wandb:
        log_wandb_images(paths)


def plot_vector_field(env, enc, forward, probe, action, name, device, paths):
    """Plot vector field for given action"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Generate grid points efficiently
    grid_x = range(5, 100, 10)
    grid_y = range(5, 100, 10)
    grid_points = np.zeros((len(grid_x) * len(grid_y), 100, 100), dtype=np.float32)
    
    idx = 0
    for x in grid_x:
        for y in grid_y:
            true_s = [x / 100.0, y / 100.0]
            grid_points[idx] = env.synth_obs(ap=true_s)
            idx += 1
    
    # Compute dynamics
    with torch.no_grad():
        xl = torch.from_numpy(grid_points).to(device)
        action_batch = torch.FloatTensor([action]).repeat(xl.size(0), 1).to(device)
        
        zt = enc(xl)
        ztn = forward(zt, action_batch)
        st = probe(zt)
        stn = probe(ztn)
    
    # Extract coordinates
    px = st[:, 0].cpu().numpy()
    py = st[:, 1].cpu().numpy()
    pu = (stn[:, 0] - st[:, 0]).cpu().numpy()
    pv = (stn[:, 1] - st[:, 1]).cpu().numpy()
    
    # Create plot
    ax.quiver(px, py, pu * 0.5, pv * 0.5, scale=1, scale_units='xy', angles='xy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Vector Field: {name} {action}")
    ax.grid(True)
    
    plt.savefig(join(paths['field_folder'], f"field_{name}.jpg"), 
                dpi=100, bbox_inches='tight')
    plt.close()


def plot_square_trajectory(env, ema_models, device, paths):
    """Plot square trajectory"""
    # Initial state
    true_s = [0.4, 0.4]
    xl = env.synth_obs(ap=true_s)
    xl = torch.from_numpy(np.array(xl, dtype=np.float32)).to(device).unsqueeze(0)
    
    zt = ema_models['enc'].ema_model(xl)
    trajectory = []
    
    # Action sequence for square
    actions = [
        [0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1],
        [-0.1, 0.0], [-0.1, 0.0], [0.0, -0.1], [0.0, -0.1],
        [0.0, 0.0], [0.0, 0.0]
    ]
    
    # Generate trajectory
    for a in actions:
        action = torch.tensor(a, dtype=torch.float32).to(device).unsqueeze(0)
        st = ema_models['a_probe'](zt)
        trajectory.append(st.cpu().numpy())
        zt = ema_models['forward'].ema_model(zt, action)
    
    trajectory = np.vstack(trajectory)
    
    # True square trajectory
    true_sq = np.array([
        [0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.6, 0.5], 
        [0.6, 0.6], [0.5, 0.6], [0.4, 0.6], [0.4, 0.5],
        [0.4, 0.4], [0.4, 0.4]
    ])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Predicted')
    ax.plot(true_sq[:, 0], true_sq[:, 1], 'm-', linewidth=2, label='True')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Square Trajectory")
    ax.legend()
    ax.grid(True)
    
    plt.savefig(join(paths['plan_folder'], "plan.jpg"), 
                dpi=100, bbox_inches='tight')
    plt.close()


def plot_distance_map(env, ema_models, dist_pred, device, paths):
    """Plot distance predictions"""
    # Starting point
    start_s = [0.45, 0.1]
    xs_obs = env.synth_obs(ap=start_s)
    xs = torch.FloatTensor(xs_obs).to(device).unsqueeze(0)
    
    # Generate grid efficiently
    grid_size = len(range(2, 99, 2)) * len(range(2, 99, 2))
    grid_points = np.zeros((grid_size, 100, 100), dtype=np.float32)
    
    idx = 0
    for x in range(2, 99, 2):
        for y in range(2, 99, 2):
            true_s = [x / 100.0, y / 100.0]
            grid_points[idx] = env.synth_obs(ap=true_s)
            idx += 1
    
    # Compute distances
    with torch.no_grad():
        xl = torch.from_numpy(grid_points).to(device)
        zl = ema_models['enc'].ema_model(xl)
        zs = ema_models['enc'].ema_model(xs)
        
        zs_expanded = zs.expand(zl.size(0), -1)
        dist_vals = dist_pred.predict_k(zl, zs_expanded)
        
        sl_probe = ema_models['a_probe'](zl).cpu().numpy()
    
    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(sl_probe[:, 0], sl_probe[:, 1], 
                         c=dist_vals.cpu().numpy(), cmap='viridis')
    plt.colorbar(scatter)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Distance Map from Starting Point")
    plt.savefig(join(paths['field_folder'], "distmap.jpg"), 
                dpi=100, bbox_inches='tight')
    plt.close()


def log_wandb_images(paths):
    """Log images to wandb"""
    image_files = {
        'fields/down': "field_down.jpg",
        'fields/up': "field_up.jpg",
        'fields/left': "field_left.jpg",
        'fields/right': "field_right.jpg",
        'fields/up-right': "field_up-right.jpg",
        'fields/plan': "../fields/plan.jpg"
    }
    
    for key, filename in image_files.items():
        img_path = join(paths['field_folder'], filename)
        if os.path.exists(img_path):
            wandb.log({key: wandb.Image(img_path)})


def save_checkpoint(models, model_path):
    """Save model checkpoint"""
    checkpoint = {name: model.state_dict() for name, model in models.items()}
    torch.save(checkpoint, model_path)


def cluster_latent(env, models, args, paths, device):
    """Cluster latent states - keeping original implementation structure"""
    # Load model
    checkpoint = torch.load(paths['model_path'], map_location=device)
    models['enc'].load_state_dict(checkpoint['enc'])
    models['a_probe'].load_state_dict(checkpoint['a_probe'])
    models['forward'].load_state_dict(checkpoint['forward'])
    
    for model in models.values():
        model.eval()
    
    cluster_trans = ClusterTransform(models['enc'], models['forward'], device)
    
    # Load dataset
    with open(paths['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loading dataset from {paths['dataset_path']}")
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
    
    # Visualize environment check
    plt.figure()
    k = 0
    grounded_traj = ast[k * 100:(k + 30) * 100]
    
    if isinstance(env, RoomPolygonObstacleEnv):
        for obs in env.obs_lst:
            x_coords, y_coords = obs.exterior.xy
            plt.plot(x_coords, y_coords, color='k')
    
    plt.plot(grounded_traj[:, 0], grounded_traj[:, 1], color='blue', linewidth=0.3)
    plt.scatter(grounded_traj[:, 0], grounded_traj[:, 1], s=2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(join(paths['logdir'], 'polygon_obstacle_detection_check.png'), dpi=600)
    plt.clf()
    
    print('Data loaded, generating latent states...')
    
    # Generate latent states with batching
    aug_latent_states = []
    predicted_grounded_states = []
    batch_size = 256
    
    for i in tqdm(range(0, X.shape[0], batch_size)):
        batch_end = min(i + batch_size, X.shape[0])
        with torch.no_grad():
            _aug_latent = cluster_trans.cluster_label_transform(
                X[i:batch_end], 
                do_augment=args.use_augmented_latent_clustering
            )
            aug_latent_states.append(_aug_latent.cpu().numpy())
            
            # Ground the states
            latent_dim = args.latent_dim
            latent_states = _aug_latent[:, :latent_dim]
            grounded = models['a_probe'](models['enc'].contrast_inv(latent_states))
            predicted_grounded_states.append(grounded.cpu().numpy())
    
    aug_latent_states = np.vstack(aug_latent_states)
    predicted_grounded_states = np.vstack(predicted_grounded_states)
    grounded_states = ast[:len(aug_latent_states)]
    
    print('Running K-means clustering...')
    kmeans = KMeans(n_clusters=args.nclusters, random_state=0, n_init=10).fit(aug_latent_states)
    predicted_labels = kmeans.labels_
    
    # Get cluster centers in grounded space
    with torch.no_grad():
        centers_latent = torch.FloatTensor(kmeans.cluster_centers_[:, :args.latent_dim]).to(device)
        centroids = models['a_probe'](models['enc'].contrast_inv(centers_latent)).cpu().numpy()
    
    # Visualize clustering
    plt.figure(figsize=(10, 10))
    plt.scatter(grounded_states[:, 0], grounded_states[:, 1], 
                c=predicted_labels, marker='.', cmap='gist_ncar', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker="*", s=200, color='black', edgecolor='white')
    
    for i, centroid in enumerate(centroids):
        plt.text(centroid[0], centroid[1], str(i), 
                horizontalalignment='center', fontsize=12)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Latent Space Clustering (k={args.nclusters})')
    plt.savefig(join(paths['field_folder'], 'latent_cluster.png'), dpi=300)
    plt.close()
    
    # Save results
    kmeans_info = {
        'kmeans': kmeans,
        'kmeans-plot': plt.gcf(),
        'grounded-cluster-center': centroids
    }
    
    with open(join(paths['logdir'], 'kmeans_info.p'), 'wb') as f:
        pickle.dump(kmeans_info, f)
    
    # Save compressed arrays
    np.savez_compressed(join(paths['data_folder'], 'latent.npz'), 
                       aug_latent_states=aug_latent_states)
    np.savez_compressed(join(paths['data_folder'], 'grounded_states.npz'), 
                       grounded_states=grounded_states)
    np.savez_compressed(join(paths['data_folder'], 'predicted_labels.npz'), 
                       predicted_labels=predicted_labels)
    
    if args.use_wandb:
        wandb.log({'latent-cluster': wandb.Image(join(paths['field_folder'], "latent_cluster.png"))})
        wandb.save(glob_str='kmeans.p', policy='now')


def generate_mdp(models, args, paths, device):
    """Generate empirical MDP from clustered states"""
    # Load models
    checkpoint = torch.load(paths['model_path'], map_location=device)
    models['enc'].load_state_dict(checkpoint['enc'])
    models['forward'].load_state_dict(checkpoint['forward'])
    
    for model in models.values():
        model.eval()
    
    cluster_trans = ClusterTransform(models['enc'], models['forward'], device)
    
    # Load clustering info
    with open(join(paths['logdir'], 'kmeans_info.p'), 'rb') as f:
        kmeans_info = pickle.load(f)
    
    kmeans = kmeans_info['kmeans']
    grounded_cluster_centers = kmeans_info['grounded-cluster-center']
    
    # Load dataset
    with open(paths['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)
    
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
    
    # Generate state labels efficiently
    print('Generating state labels...')
    states_label = []
    batch_size = 256
    
    for i in tqdm(range(0, len(X), batch_size)):
        batch_end = min(i + batch_size, len(X))
        with torch.no_grad():
            _aug_latent = cluster_trans.cluster_label_transform(
                X[i:batch_end], 
                do_augment=args.use_augmented_latent_clustering
            )
            labels = kmeans.predict(_aug_latent.cpu().numpy())
            states_label.extend(labels)
    
    states_label = np.array(states_label[:-1])
    next_state = np.array(states_label[1:])
    
    # Filter for small actions
    A_filtered = A[:-1]
    action_mask = np.abs(A_filtered).sum(1) < 0.1
    
    states_label = states_label[action_mask]
    next_state = next_state[action_mask]
    A_filtered = A_filtered[action_mask]
    
    print(f'Filtered data shapes: states={states_label.shape}, actions={A_filtered.shape}')
    
    # Create empirical MDP
    empirical_mdp = EmpiricalMDP(
        state=states_label,
        action=A_filtered,
        next_state=next_state,
        reward=np.zeros_like(A_filtered)
    )
    
    # Visualize transitions
    plt.figure(figsize=(10, 10))
    for cluster_i, cluster_center in enumerate(tqdm(grounded_cluster_centers)):
        if cluster_i not in empirical_mdp.unique_states_dict:
            continue
        
        state_idx = empirical_mdp.unique_states_dict[cluster_i]
        for next_idx, action in enumerate(empirical_mdp.transition[state_idx]):
            if not np.isnan(action).all():
                plt.quiver(cluster_center[0], cluster_center[1], 
                          action[0], action[1], alpha=0.7)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Latent Cluster Transitions')
    plt.savefig(join(paths['field_folder'], 'latent_cluster_with_action_vector.png'))
    plt.close()
    
    # Save transition visualization
    empirical_mdp.visualize_transition(save_path=join(paths['field_folder'], 'transition_img.png'))
    
    # Save MDP
    with open(join(paths['logdir'], 'empirical_mdp.p'), 'wb') as f:
        pickle.dump(empirical_mdp, f)
    
    if args.use_wandb:
        wandb.log({'mdp': wandb.Image(join(paths['field_folder'], "transition_img.png"))})
        wandb.save(glob_str='empirical_mdp.p', policy="now")


def high_low_plan(env, models, args, paths, device):
    """High-low hierarchical planning"""
    # Load models
    checkpoint = torch.load(paths['model_path'], map_location=device)
    for name in ['enc', 'forward', 'a_probe']:
        models[name].load_state_dict(checkpoint[name])
        models[name].eval()
    
    # Setup dynamics wrapper
    dynamics = LatentWrapper(models['forward'])
    dynamics.eval()
    
    cluster_trans = ClusterTransform(models['enc'], models['forward'], device)
    
    # Load empirical MDP and clustering
    with open(join(paths['logdir'], 'empirical_mdp.p'), 'rb') as f:
        empirical_mdp = pickle.load(f)
    
    with open(join(paths['logdir'], 'kmeans_info.p'), 'rb') as f:
        kmeans_info = pickle.load(f)
    
    kmeans = kmeans_info['kmeans']
    grounded_cluster_centers = kmeans_info['grounded-cluster-center']
    
    # Load dataset
    with open(paths['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)
    
    X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']
    
    # Filter dataset
    X = X[:-1]
    ast = ast[:-1]
    A = A[:-1]
    
    action_mask = np.abs(A).sum(1) < 0.1
    X = X[action_mask]
    ast = ast[action_mask]
    A = A[action_mask]
    
    # Setup planning
    exp_id = args.exp_id
    traj_opt_fig_dir = join(paths['logdir'], 'high_low_data')
    os.makedirs(traj_opt_fig_dir, exist_ok=True)
    
    # Get start and goal states
    from_to = args.from_to if isinstance(args.from_to, list) else [13, 2]
    
    init_mdp_state = from_to[0]
    init_obs, init_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, init_mdp_state)
    init_obs_tensor = torch.from_numpy(np.array(init_obs, dtype=np.float32)).to(device).unsqueeze(0)
    init_lat_state = models['enc'](init_obs_tensor)
    
    target_mdp_state = from_to[1]
    target_obs, target_gt_agent_state = obs_sampler(X, ast, empirical_mdp.state, target_mdp_state)
    target_obs_tensor = torch.from_numpy(np.array(target_obs, dtype=np.float32)).to(device).unsqueeze(0)
    target_lat_state = models['enc'](target_obs_tensor)
    
    nz = init_lat_state.size(1)
    nu = ast.shape[1]
    
    # Execute planning
    from traj_opt.room_planner import room_high_low_planner, room_low_planner
    
    # High-low planning
    traj_opt_data_path = join(traj_opt_fig_dir, f'high_low_{exp_id}.p')
    high_low_mpc_data = room_high_low_planner(
        env, nz, nu, models['enc'], dynamics, models['a_probe'], 
        empirical_mdp, kmeans, cluster_trans,
        init_gt_agent_state, init_lat_state, init_mdp_state,
        target_gt_agent_state, target_lat_state, target_mdp_state,
        save_path=traj_opt_data_path, 
        augmented=args.use_augmented_latent_clustering
    )
    
    # Low-level planning
    traj_opt_data_path = join(traj_opt_fig_dir, f'low_{exp_id}.p')
    low_mpc_data = room_low_planner(
        env, nz, nu, models['enc'], dynamics, models['a_probe'], 
        empirical_mdp, kmeans, cluster_trans,
        init_gt_agent_state, init_lat_state, init_mdp_state,
        target_gt_agent_state, target_lat_state, target_mdp_state,
        save_path=traj_opt_data_path, 
        augmented=args.use_augmented_latent_clustering
    )
    
    # Report runtimes
    high_low_runtime = np.mean(high_low_mpc_data['mpc_time'])
    low_runtime = np.mean(low_mpc_data['mpc_time'])
    print(f'Avg. runtime - High-low: {high_low_runtime:.3f}s, Low: {low_runtime:.3f}s')
    
    # Visualize results
    plot_planning_results(env, high_low_mpc_data, low_mpc_data, 
                         target_gt_agent_state, traj_opt_fig_dir, exp_id)


def plot_planning_results(env, high_low_data, low_data, target_state, 
                         save_dir, exp_id):
    """Plot planning results"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles
    for obstacle in env.obs_lst:
        x, y = obstacle.exterior.xy
        plt.plot(x, y, 'k-', linewidth=2)
    
    # Plot trajectories
    high_low_traj = high_low_data['grounded_states']
    low_traj = low_data['grounded_states']
    
    plt.plot(high_low_traj[:, 0], high_low_traj[:, 1], 'b-', 
             linewidth=3, label='High+Low')
    plt.scatter(high_low_traj[:, 0], high_low_traj[:, 1], c='b', s=30)
    
    plt.plot(low_traj[:, 0], low_traj[:, 1], 'g-.', 
             linewidth=3, label='Low Only')
    plt.scatter(low_traj[:, 0], low_traj[:, 1], c='g', s=30)
    
    # Mark start and goal
    plt.scatter(high_low_traj[0, 0], high_low_traj[0, 1], 
                marker='o', s=100, c='k', label='Start')
    plt.scatter(target_state[0], target_state[1], 
                marker='s', s=100, c='r', label='Goal')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(join(save_dir, f'{exp_id}.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved planning results to {join(save_dir, f'{exp_id}.png')}")


if __name__ == '__main__':
    main()