#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/validation results/benchmark

echo "Compiling experimental striped OpenMP variant ..."
gcc -O3 -g -Werror -fopenmp sandpile_striped_openmp.c -lm -o sandpile_striped.exe

for n in 101 201 401; do
    echo "Checking N=$n against the reference ..."
    ./sandpile.exe "$n" > /dev/null
    cp identity.png "results/validation/striped_ref_${n}.png"
    OMP_NUM_THREADS=4 ./sandpile_striped.exe "$n" > /dev/null
    cmp "results/validation/striped_ref_${n}.png" identity.png
    echo "N=$n: striped output MATCH"
done

if [[ "${FULL:-0}" == "1" ]]; then
    ref="results/baseline/identity_1001.png"
    if [[ ! -f "$ref" ]]; then
        echo "Missing original N=1001 baseline: $ref" >&2
        exit 1
    fi
    echo "Benchmarking striped OpenMP at N=1001 on 4 threads ..."
    OMP_NUM_THREADS=4 /usr/bin/time -v -o results/benchmark/striped_1001.time ./sandpile_striped.exe 1001
    cmp "$ref" identity.png
    cp identity.png results/benchmark/striped_1001.png
    echo "N=1001: striped output MATCH"
    grep -E 'Elapsed \(wall clock\)|User time|Maximum resident set size' results/benchmark/striped_1001.time
else
    echo "Small/medium checks passed. To test N=1001: FULL=1 bash test_hw3_striped.sh"
fi
