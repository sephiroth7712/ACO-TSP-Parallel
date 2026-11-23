#!/bin/bash
#SBATCH --error ACO_TSP_Serial_error.log
#SBATCH --output ACO_TSP_Serial_output.log
#SBATCH --job-name=ACO_TSP_Serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

export file_name=datasets/a280.tsp
export iterations=2000
export n_ants=100
export alpha=1
export beta=5.5
export ro=0.1
export tau_max=2
export tau_min=0.1
export q=2590

./ACO_TSP_Serial $file_name $n_ants $iterations $alpha $beta $ro $tau_max $tau_min $q