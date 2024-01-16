#!/bin/bash

set -e

N_PROC=${1:-"4"}
CLEANUP=${2:-"false"}

OUT_DIR="/tmp/solve_many_linsys"

if [ "${CLEANUP}" = "true" ]; then
    cargo clean --release --target-dir $OUT_DIR
fi

cargo build --release --target-dir $OUT_DIR

BIN=$OUT_DIR/release/simple_mpi

mpiexec --oversubscribe -np $N_PROC $BIN -- $GRID_NX $GENIE
