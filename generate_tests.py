#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point

from room_polygon_obstacle_env import RoomPolygonObstacleEnv, RoomMultiPassageEnv, RoomSpiral, RoomMultiPassageEnvLarge

def get_env(env_name):
    if env_name == 'polygon-obs':
        return RoomPolygonObstacleEnv()
    elif env_name == 'room-multi-passage':
        return RoomMultiPassageEnv()
    elif env_name == 'room-spiral':
        return RoomSpiral()
    elif env_name == 'room-multi-passage-large':
        return RoomMultiPassageEnvLarge()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def is_valid_position(pos, env):
    point = Point(pos[0], pos[1])
    for obs in env.obs_lst:
        if obs.contains(point):
            return False
    return True

def generate_random_valid_positions(env, num_pairs=50, seed=42, min_distance=0.3):
    np.random.seed(seed)
    random.seed(seed)
    
    valid_pairs = []
    max_attempts = 1000
    
    for i in tqdm(range(num_pairs), desc="Generating test cases"):
        attempts = 0
        while attempts < max_attempts:
            start_pos = [random.uniform(0.0, 0.99), random.uniform(0.0, 0.99)]
            goal_pos = [random.uniform(0.0, 0.99), random.uniform(0.0, 0.99)]
            
            # Check if positions are valid and sufficiently far apart
            distance = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if (is_valid_position(start_pos, env) and 
                is_valid_position(goal_pos, env) and
                distance > min_distance):
                valid_pairs.append({
                    'test_id': i,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'distance': distance
                })
                break
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not find valid pair {i}, using fallback")
            # Fallback to known valid positions with some variation
            start_pos = [0.1 + (i % 5) * 0.05, 0.1 + (i % 5) * 0.05]
            goal_pos = [0.9 - (i % 5) * 0.05, 0.9 - (i % 5) * 0.05]
            valid_pairs.append({
                'test_id': i,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'distance': np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            })
    
    return valid_pairs

def main():
    parser = argparse.ArgumentParser(description='Generate fixed test set for planner evaluation')
    parser.add_argument('--env', required=True, 
                        choices=['polygon-obs', 'room-multi-passage', 'room-spiral', 'room-multi-passage-large'],
                        help='Environment type')
    parser.add_argument('--logdir', default='logs', help='Base log directory')
    parser.add_argument('--num-episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--seed', default=42, type=int, help='Seed for test set generation')
    parser.add_argument('--min-distance', default=0.2, type=float, help='Minimum distance between start and goal')
    
    args = parser.parse_args()
    
    # Create environment
    env = get_env(args.env)
    
    # Create directories
    eval_dir = os.path.join(args.logdir, args.env, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate test cases
    test_set = generate_random_valid_positions(
        env, 
        num_pairs=args.num_episodes, 
        seed=args.seed,
        min_distance=args.min_distance
    )
    
    # Add metadata
    test_set_data = {
        'env_type': args.env,
        'num_episodes': args.num_episodes,
        'generation_seed': args.seed,
        'min_distance': args.min_distance,
        'test_cases': test_set,
        'stats': {
            'avg_distance': np.mean([case['distance'] for case in test_set]),
            'min_distance': np.min([case['distance'] for case in test_set]),
            'max_distance': np.max([case['distance'] for case in test_set]),
            'std_distance': np.std([case['distance'] for case in test_set])
        }
    }
    
    # Save test set
    test_set_path = os.path.join(eval_dir, f'test_set_{args.env}.p')
    pickle.dump(test_set_data, open(test_set_path, 'wb'))

if __name__ == "__main__":
    main()