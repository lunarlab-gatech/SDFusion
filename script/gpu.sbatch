#!/bin/bash
#SBATCH -Jgpujob                               # Job name
#SBATCH --account=gts-lgan31                    # charge account
#SBATCH -N1 --gres=gpu:V100:1                   # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=24G                       # Memory per core
#SBATCH -t60                                    # Duration of the job (Ex: t15: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -ooutput/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=szhao381@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
module load cuda/11.6.0-u4jzhg                  # Load module dependencies

conda activate sdfusion_cu116                   # Conda environment name

srun python mm2shape.py