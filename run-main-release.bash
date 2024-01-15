#!/bin/bash

set -e

CLEANUP=${1:-""}

OUT_DIR="/tmp/solve_many_linsys"

if [ "${CLEANUP}" = "1" ]; then
    cargo clean --release --target-dir $OUT_DIR
fi

cargo build --release --target-dir $OUT_DIR

mpiexec -np 4 $OUT_DIR/release/simple_laplacian
