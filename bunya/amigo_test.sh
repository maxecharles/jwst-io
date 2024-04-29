#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --job-name=amigo
#SBATCH --time=0:01:00
#SBATCH --partition=general
#SBATCH --account=a_astro
#SBATCH --output=results/visfit.out
#SBATCH --error=results/visfit.error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=max.charles@sydney.edu.au

# Load the necessary modules
module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate dlux

# Print the environemnts
pip install --upgrade --no-deps --force-reinstall git+ssh://git@github.com/LouisDesdoigts/amigo.git@dev -q
pip install git+https://git@github.com/itroitskaya/dLuxWebbpsf.git@import_fix -q
pip install git+https://git@github.com/fmartinache/xara.git -q

# Run the python script
srun --unbuffered python /scratch/user/uqmchar4/code/jwst-io/amigo_test.py >> output/amigo_test.out