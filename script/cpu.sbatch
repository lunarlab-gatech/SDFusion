#!/bin/bash
#SBATCH -Jcpujob                               # Job name
#SBATCH --account=gts-lgan31                    # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes
#SBATCH --mem-per-cpu=12G                       # Memory per core
#SBATCH -t60                                    # Duration of the job (Ex: t60: 60 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -ooutput/Report-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=szhao381@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory


source ./isosurface/LIB_PATH
dset='shapenet'
reduce=4
category='chair'

module load anaconda3/2022.05                   # Load module dependencies
conda activate sdfusion_cu116                   # Conda environment name

srun python -u create_sdf.py --dset ${dset} --thread_num 9 --reduce ${reduce} --category ${category}