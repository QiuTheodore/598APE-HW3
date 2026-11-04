#!/usr/bin/env bash
# New experimental implementation: never overwrites prior optimized sources.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/validation results/benchmark

if [[ ! -x ./sandpile.exe ]]; then
    echo "Missing original ./sandpile.exe; compile baseline first: make all" >&2
    exit 1
fi

echo "Compiling symmetry-aware OpenMP experiment (GCC -O3) ..."
gcc -O3 -g -Werror -fopenmp sandpile_symmetry_openmp.c -lm -o sandpile_symmetry.exe

# On a 4-vCPU teaching VM, keep thread count fixed and avoid dynamic changes.
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE

for n in 20 101 201 401; do
    echo "Checking N=$n against the original reference ..."
    ./sandpile.exe "$n" >/dev/null
    cp identity.png "results/validation/symmetry_ref_${n}.png"
    ./sandpile_symmetry.exe "$n" >/dev/null
    if ! cmp -s "results/validation/symmetry_ref_${n}.png" identity.png; then
        echo "N=$n: MISMATCH; do not use this version" >&2
        exit 1
    fi
    echo "N=$n: symmetry output MATCH"
done

if [[ "${FULL:-0}" == "1" ]]; then
    ref="results/baseline/identity_1001.png"
    if [[ ! -f "$ref" ]]; then
        echo "Missing saved baseline $ref; stop rather than overwrite baseline" >&2
        exit 1
    fi
    echo "Benchmarking symmetry-aware OpenMP, N=1001, 4 threads ..."
    /usr/bin/time -v -o results/benchmark/symmetry_1001.time ./sandpile_symmetry.exe 1001
    if ! cmp -s "$ref" identity.png; then
        echo "N=1001: MISMATCH; do not use this version" >&2
        exit 1
    fi
    cp identity.png results/benchmark/symmetry_1001.png
    echo "N=1001: symmetry output MATCH"
    echo "RESULTS:"
    grep -E 'User time|Elapsed \(wall clock\)|Maximum resident set size' results/benchmark/symmetry_1001.time
else
    echo "Small/medium tests passed. For N=1001: FULL=1 bash test_hw3_symmetry.sh"
fi
