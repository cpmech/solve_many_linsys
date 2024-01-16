#!/bin/bash

set -e

N_PROC=${1:-"4"}
GRID_NX=${2:-"21"}
GENIE=${3:-"umfpack"}
CLEANUP=${4:-"false"}

OUT_DIR="/tmp/solve_many_linsys"

if [ "${CLEANUP}" = "true" ]; then
    cargo clean --release --target-dir $OUT_DIR
fi

cargo build --release --target-dir $OUT_DIR

BIN=$OUT_DIR/release/mpi_heat_equation

mpiexec --oversubscribe -np $N_PROC $BIN -- $GRID_NX $GENIE
