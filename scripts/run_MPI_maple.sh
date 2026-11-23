#!/bin/bash
file_name=datasets/eil51.tsp
iterations=2000;
n_ants=50;
alpha=1;
beta=3.5;
ro=0.1;
tau_max=2;
tau_min=0.1;
q=2587
rm log_mpi.txt
mpirun -oversubscribe -np 4 ./ACO_TSP_MPI $file_name $n_ants $iterations $alpha $beta $ro $tau_max $tau_min $q > log_mpi.txt