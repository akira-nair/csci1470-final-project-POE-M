#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 0:05:00
cd /users/anair27/data/anair27/misc/dl-final/csci1470-final-project-POE-M/code
sbatch train_model_line_by_line.sh 1
sbatch train_model_line_by_line.sh 2
sbatch train_model_line_by_line.sh 3