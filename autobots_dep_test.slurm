#!/bin/bash
#SBATCH --job-name=AutobotsDepTest   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jwille@princeton.edu
#SBATCH --output=/home/jwille/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/home/jwille/slurm_output/%x.%j.err  # STDERR file

module purge
module load anaconda3/2022.10
conda activate AutoBots

python dep_test.py