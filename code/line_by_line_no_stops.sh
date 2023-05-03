#!/bin/sh
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o line_by_line_nostops_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
# conda init bash
# conda activate /users/anair27/anaconda/akira_conda
cd /users/anair27/data/anair27/misc/dl-final/csci1470-final-project-POE-M/code
python3 line_by_line_no_stops.py ${1}