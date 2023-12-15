#!/bin/bash
#SBATCH -Jtestenv                               # Job name
#SBATCH --account=gts-lgan31                    # charge account
#SBATCH -N1 --gres=gpu:V100:1                   # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                        # Memory per core
#SBATCH -t10                                    # Duration of the job (Ex: 10 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=szhao381@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
module load cuda/12.1.1-6oacj6                  # Load module dependencies

conda activate sdfusion                         # Conda environment name

srun python test_env.py