#!/bin/bash
#SBATCH --job-name=AutobotsSubmitNuscenesEval   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # number of processes
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jwille@princeton.edu
#SBATCH --output=/home/jwille/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/home/jwille/slurm_output/%x.%j.err  # STDERR file

module purge
module load anaconda3/2022.10
conda activate AutoBots

python generate_nuscene_results.py --dataset-path h5files --models-path results/Nuscenes/Autobot_joint_C10_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_roadLanes_test_s1/models_140.pth