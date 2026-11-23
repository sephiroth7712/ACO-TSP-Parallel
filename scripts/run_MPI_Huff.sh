#!/bin/bash
#SBATCH --error ACO_TSP_MPI_error.log
#SBATCH --output ACO_TSP_MPI_output.log
#SBATCH --job-name=ACO_TSP_MPI
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=25
#SBATCH --nodelist=huff27
#SBATCH --mem=8G

export file_name=datasets/a280.tsp
export iterations=2000
export n_ants=50
export alpha=1
export beta=5.5
export ro=0.1
export tau_max=2
export tau_min=0.1
# export q=430
export q=2590

mpirun -np 2 ./ACO_TSP_MPI $file_name $n_ants $iterations $alpha $beta $ro $tau_max $tau_min $q