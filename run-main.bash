#!/bin/bash

set -e

NP=4
if [[ "$CI" == "true" ]]; then
    NP=2
fi

export CARGO_TARGET_DIR="/tmp/solve_many_linsys"
BINDIR="/tmp/solve_many_linsys/release/"

cargo build --release

mpiexec -np $NP $BINDIR/solve_many_linsys
