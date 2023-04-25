#!/bin/bash
#SBATCH --job-name=GenerateNuscenesValH5File    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # number of processes
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jwille@princeton.edu
#SBATCH --output=/home/jwille/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/home/jwille/slurm_output/%x.%j.err  # STDERR file

. /etc/profile

module purge
module load anaconda3/2022.10
conda activate AutoBots

python create_h5_nusc.py --raw-dataset-path /scratch/gpfs/jwille/v1.0-trainval_full --split-name val --output-h5-path /scratch/gpfs/jwille/AutoBots/h5files/
