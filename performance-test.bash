#!/bin/bash

set -e

bash run-mpi-poisson.bash 1 1000 umfpack true

# for i in $(seq 2 24); do
for i in 2 4 8 16 24; do
    bash run-mpi-poisson.bash $i 1000 umfpack
done

