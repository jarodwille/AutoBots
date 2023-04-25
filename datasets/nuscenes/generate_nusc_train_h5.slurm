#!/bin/bash
#SBATCH --job-name=GenerateNuscenesTrainH5File # create a short name for your job
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

. /etc/profile

module purge
module load anaconda3/2022.10
conda activate AutoBots

python create_h5_nusc.py --raw-dataset-path /scratch/gpfs/jwille/v1.0-trainval_full --split-name train --output-h5-path /scratch/gpfs/jwille/AutoBots/h5files/
