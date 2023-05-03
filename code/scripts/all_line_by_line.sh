#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 0:05:00
#SBATCH -o None
cd /users/anair27/data/anair27/misc/dl-final/csci1470-final-project-POE-M/code
# sbatch line_by_line_no_stops.sh 1
# sbatch line_by_line_no_stops.sh 2
# sbatch line_by_line_no_stops.sh 3
sbatch scripts/line_by_line_with_stops.sh 1
sbatch scripts/line_by_line_with_stops.sh 2
sbatch scripts/line_by_line_with_stops.sh 3