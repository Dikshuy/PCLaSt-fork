import torch

from .high_level_planner.dijkstra_planner import Dijkstra_Planner
from .hj_prox.hj_prox_alg import Tracking_Cost, HJ_Prox_Optimizer
import copy
from tqdm import tqdm
import time
import numpy as np

from .cem.cem_optimizer import CEM_Optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

method = 'cem'

class LowPlannerParams:
    def __init__(self) -> None:
        self.n_batch = 1
        self.T = 30
        self.horizon = 10
        self.u_min = -0.2
        self.u_max = 0.2
        self.t_param = 0.1
        self.iter_num = 10
        
        self.cem_samples = 500

def room_high_low_planner(env, nz, nu, enc, dynamics, a_probe, empirical_mdp, kmeans, cluster_trans,
                          init_gt_agent_state, init_lat_state, init_mdp_state, 
                          target_gt_agent_state, target_lat_state, target_mdp_state,
                          save_path = None, augmented = False, success_threshold = 0.030):
    # use both high-level and low-level planner in the room env. 

    opt_params = LowPlannerParams()

    n_batch, T, horizon = opt_params.n_batch, opt_params.T, opt_params.horizon
    u_min, u_max = opt_params.u_min, opt_params.u_max
    
    dijkstra_planner = Dijkstra_Planner(empirical_mdp)
    tracking_cost_fcn = Tracking_Cost(dynamics, init_lat_state, target_lat_state)
    
    init_actions = u_min + (u_max - u_min) * torch.rand((n_batch, horizon, nu)).to(device)
 
    # closed-loop simulation
    lat_state_log = torch.zeros((T + 1, 1, nz)).to(device)
    control_log = torch.zeros((T, 1, nu)).to(device)

    gt_agent_state_log = np.zeros((T + 1, init_gt_agent_state.shape[0]))
    probe_lat_state_log = torch.zeros((T + 1, init_gt_agent_state.shape[0])).to(device)
    
    executed_mdp_states = []
    planned_mdp_states = []

    z_t = init_lat_state
    lat_state_log[0] = z_t
    gt_agent_state_log[0] = copy.deepcopy(init_gt_agent_state)
    probe_lat_state_log[0] = a_probe(init_lat_state)

    mpc_time = []
    gt_agent_state = copy.deepcopy(init_gt_agent_state)

    t_param = opt_params.t_param
    iter_num = opt_params.iter_num
    actual_steps = 0
    
    for t in tqdm(range(T), desc='high_low_mpc'):
        # Check if goal reached (early termination)
        current_distance = np.linalg.norm(gt_agent_state - target_gt_agent_state)
        if current_distance <= success_threshold:
            print(f"Goal reached at step {t}! Distance: {current_distance:.4f}")
            actual_steps = t
            break
            
        start_time = time.time()

        # call high level planner
        aug_z_t = cluster_trans.cluster_label_transform_latent(z_t, do_augment=augmented)
        current_mdp_state = kmeans.predict(aug_z_t.detach().cpu())[0]
        executed_mdp_states.append(current_mdp_state)

        if current_mdp_state != target_mdp_state:
            next_mdp_state = dijkstra_planner.step(current_mdp_state, target_mdp_state)
            planned_mdp_states.append(next_mdp_state)

            next_lat_state = kmeans.cluster_centers_[next_mdp_state][:256]
            next_lat_state = torch.FloatTensor(next_lat_state).unsqueeze(0).to(device)
            next_lat_state = enc.contrast_inv(next_lat_state)
        else:
            next_lat_state = target_lat_state
            planned_mdp_states.append(target_mdp_state)

        tracking_cost_fcn = Tracking_Cost(dynamics, z_t, next_lat_state)

        # hj prox
        if method == 'hj-prox':
            hj_optimizer = HJ_Prox_Optimizer(tracking_cost_fcn, init_actions, t_param, x_min=u_min, x_max=u_max)
            output_action, action_list = hj_optimizer.prox_grad_descent(iter_num = iter_num, x_init = init_actions, t = t_param, beta = t_param)

        # cem
        if method == 'cem':
            cem_optimizer = CEM_Optimizer(tracking_cost_fcn, x_min = u_min, x_max = u_max)
            output_action, _ = cem_optimizer.cem_iter(init_actions, num_samples=opt_params.cem_samples, num_iter=opt_params.iter_num)

        rollout_costs = tracking_cost_fcn(output_action)
        selected_action = output_action[rollout_costs[0].argmin().item()]

        action = selected_action[0:1, :]

        control_log[t] = action

        print(f'agent state: {gt_agent_state}, goal state: {target_gt_agent_state}')
        print(f'action: {action}')

        env.agent_pos = gt_agent_state
        
        env.step(action[0].detach().cpu().numpy())
        next_obs, next_agent_pos, _ = env.get_obs()

        gt_agent_state_log[t + 1] = copy.deepcopy(next_agent_pos)
        gt_agent_state = next_agent_pos

        next_lat_state = enc(torch.FloatTensor(next_obs).unsqueeze(0).to(device))
        z_t = next_lat_state

        lat_state_log[t + 1] = z_t

        # TODO: update u_init in an adaptive manner
        init_actions = u_min + (u_max - u_min) * torch.rand((n_batch, horizon, nu)).to(device)
        init_actions[0, :] = torch.cat((selected_action[1:, :], torch.zeros((1, nu)).to(device)), dim=0)

        run_time = time.time() - start_time
        mpc_time.append(run_time)
        actual_steps = t + 1

    print(f'simulation steps: {actual_steps}, total runtime: {sum(mpc_time):.2f}s')

    # trim logs to actual length
    lat_state_log = lat_state_log[:actual_steps+1]
    control_log = control_log[:actual_steps]
    gt_agent_state_log = gt_agent_state_log[:actual_steps+1]

    mpc_data = {'grounded_states': gt_agent_state_log, 'actions': control_log, 'mpc_time': mpc_time,
                'target_grounded_state': target_gt_agent_state, 'lat_state_log': lat_state_log,
                'executed_mdp_states': executed_mdp_states, 'planned_mdp_states': planned_mdp_states}

    if save_path is not None:
        torch.save(mpc_data, save_path)

    return mpc_data

def room_low_planner(env, nz, nu, enc, dynamics, a_probe, empirical_mdp, kmeans, cluster_trans,
                    init_gt_agent_state, init_lat_state, init_mdp_state, 
                    target_gt_agent_state, target_lat_state, target_mdp_state,
                    save_path = None, augmented = False, success_threshold = 0.030):
    # use both high-level and low-level planner in the room env. 

    opt_params = LowPlannerParams()

    n_batch, T, horizon = opt_params.n_batch, opt_params.T, opt_params.horizon
    u_min, u_max = opt_params.u_min, opt_params.u_max
    
    dijkstra_planner = Dijkstra_Planner(empirical_mdp)
    tracking_cost_fcn = Tracking_Cost(dynamics, init_lat_state, target_lat_state)
    
    init_actions = u_min + (u_max - u_min) * torch.rand((n_batch, horizon, nu)).to(device)
 
    # closed-loop simulation
    lat_state_log = torch.zeros((T + 1, 1, nz)).to(device)
    control_log = torch.zeros((T, 1, nu)).to(device)

    gt_agent_state_log = np.zeros((T + 1, init_gt_agent_state.shape[0]))
    probe_lat_state_log = torch.zeros((T + 1, init_gt_agent_state.shape[0])).to(device)

    z_t = init_lat_state
    lat_state_log[0] = z_t
    gt_agent_state_log[0] = copy.deepcopy(init_gt_agent_state)
    probe_lat_state_log[0] = a_probe(init_lat_state)

    mpc_time = []
    gt_agent_state = copy.deepcopy(init_gt_agent_state)

    t_param = opt_params.t_param
    iter_num = opt_params.iter_num
    actual_steps = 0
    
    for t in tqdm(range(T), desc='low_mpc'):
        # Check if goal reached (early termination)
        current_distance = np.linalg.norm(gt_agent_state - target_gt_agent_state)
        if current_distance <= success_threshold:
            print(f"Goal reached at step {t}! Distance: {current_distance:.4f}")
            actual_steps = t
            break
            
        start_time = time.time()

        tracking_cost_fcn = Tracking_Cost(dynamics, z_t, target_lat_state)

        # hj prox
        if method == 'hj-prox':
            hj_optimizer = HJ_Prox_Optimizer(tracking_cost_fcn, init_actions, t_param, x_min=u_min, x_max=u_max)
            output_action, action_list = hj_optimizer.prox_grad_descent(iter_num = iter_num, x_init = init_actions, t = t_param, beta = t_param)

        # cem
        if method == 'cem':
            cem_optimizer = CEM_Optimizer(tracking_cost_fcn, x_min = u_min, x_max = u_max)
            output_action, _ = cem_optimizer.cem_iter(init_actions, num_samples=opt_params.cem_samples, num_iter=opt_params.iter_num)

        rollout_costs = tracking_cost_fcn(output_action)
        selected_action = output_action[rollout_costs[0].argmin().item()]

        action = selected_action[0:1, :]

        control_log[t] = action

        print(f'agent state: {gt_agent_state}, goal state: {target_gt_agent_state}')
        print(f'action: {action}')

        env.agent_pos = gt_agent_state
        env.step(action[0].detach().cpu().numpy())
        next_obs, next_agent_pos, _ = env.get_obs()

        gt_agent_state_log[t + 1] = copy.deepcopy(next_agent_pos)
        gt_agent_state = next_agent_pos

        next_lat_state = enc(torch.FloatTensor(next_obs).unsqueeze(0).to(device))
        z_t = next_lat_state

        lat_state_log[t + 1] = z_t

        init_actions = u_min + (u_max - u_min) * torch.rand((n_batch, horizon, nu)).to(device)
        init_actions[0, :] = torch.cat((selected_action[1:, :], torch.zeros((1, nu)).to(device)), dim=0)

        run_time = time.time() - start_time
        mpc_time.append(run_time)
        actual_steps = t + 1

    print(f'simulation steps: {actual_steps}, total runtime: {sum(mpc_time):.2f}s')

    # Trim logs to actual length
    lat_state_log = lat_state_log[:actual_steps+1]
    control_log = control_log[:actual_steps]
    gt_agent_state_log = gt_agent_state_log[:actual_steps+1]

    mpc_data = {'grounded_states': gt_agent_state_log, 'actions': control_log, 'mpc_time': mpc_time,
                'target_grounded_state': target_gt_agent_state, 'lat_state_log': lat_state_log}

    if save_path is not None:
        torch.save(mpc_data, save_path)

    return mpc_data