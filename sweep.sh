#!/bin/bash
#SBATCH --account=rrg-mtaylor3
#SBATCH --mem-per-cpu=128G
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --job-name=pclast_multi_env
#SBATCH --output=logs/pclast_env_%a.out
#SBATCH --error=logs/pclast_env_%a.err

module load python
module load httpproxy
source .venv/bin/activate
pip install --no-index wandb
wandb login $API_KEY

ENVS=(
    "room-multi-passage"
    "polygon-obs"
    "room-spiral"
)

NUM_SEEDS=10
env_index=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
seed=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

env=${ENVS[$env_index]}

echo "Starting job for environment: $env, seed: $seed"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Environment index: $env_index"
echo "Environment: $env"
echo "Seed: $seed"

mkdir -p logs/seed_${seed}
logdir="logs/seed_${seed}"

echo "Log directory: $logdir"

# unique wandb run name
export WANDB_RUN_NAME="${env}_seed_${seed}_job_${SLURM_ARRAY_TASK_ID}"

set -e

echo "================================================================================================="

python main.py --env ${env} --opr generate-data --seed ${seed} --logdir ${logdir}
python main.py --env ${env} --opr train --max_k 10 --contrastive --contrastive_k 2 --ndiscrete 64 --seed ${seed} --logdir ${logdir}
python main.py --env ${env} --opr cluster-latent --seed ${seed} --logdir ${logdir}
python main.py --env ${env} --opr generate-mdp --seed ${seed} --logdir ${logdir}
python main.py --env ${env} --opr high-low-plan --from_to 2 15 --seed ${seed} --logdir ${logdir}
python main.py --env ${env} --opr evaluate-planners --seed ${seed} --logdir ${logdir}

echo "================================================================================================="
echo "Completed pipeline for environment: $env, seed: $seed"
