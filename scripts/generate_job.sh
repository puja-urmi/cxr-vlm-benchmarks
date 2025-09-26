#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-01:00:00
#SBATCH --output=logs_%J.log   

# Load the required modules
module load python/3.11
source /home/psaha03/scratch/env/bin/activate

python /home/psaha03/scratch/xray-report-generation/scripts/generate_tuned_cxr.py