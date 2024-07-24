#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --job-name=regs
#SBATCH --time=0:03:00
#SBATCH --partition=general
#SBATCH --account=a_astro
#SBATCH --output=/scratch/user/uqmchar4/code/jwst-io/bunya/outputs/io_optim.out
#SBATCH --error=/scratch/user/uqmchar4/code/jwst-io/bunya/outputs/io_optim.error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uqmchar4@uq.edu.au

# Load the necessary modules
module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate amigo

# Print the environments
pip install --upgrade --no-deps --force-reinstall git+ssh://git@github.com/LouisDesdoigts/amigo.git@dev -q
pip install git+https://git@github.com/itroitskaya/dLuxWebbpsf.git@import_fix -q
pip install git+https://git@github.com/fmartinache/xara.git -q

# Run the python script
srun --unbuffered python /scratch/user/uqmchar4/code/jwst-io/reg_grid.py 5 >> /scratch/user/uqmchar4/code/jwst-io/bunya/outputs/io_optim.out