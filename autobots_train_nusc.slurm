#!/bin/bash
#SBATCH --job-name=AutobotTrainNusc    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # number of processes
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jwille@princeton.edu
#SBATCH --output=/home/jwille/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/home/jwille/slurm_output/%x.%j.err  # STDERR file

. /etc/profile

module purge
module load anaconda3/2022.10
conda activate AutoBots

python train.py \
    --exp-id test \
    --seed 1 \
    --dataset Nuscenes --model-type Autobot-Joint --num-modes 10 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 128 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /scratch/gpfs/jwille/h5files/
