#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/baseline results/validation results/benchmark

# Save the original 1001x1001 image BEFORE running this script:
#   cp identity.png results/baseline/identity_1001.png
if [[ ! -f results/baseline/identity_1001.png ]]; then
  echo 'ERROR: save the original 1001x1001 baseline PNG first:' >&2
  echo 'cp identity.png results/baseline/identity_1001.png' >&2
  exit 1
fi

printf 'Compiling with GCC -O3 ...\n'
gcc -O3 -g -Werror sandpile_optimized.c -lm -o sandpile_opt.exe
gcc -O3 -g -Werror -fopenmp sandpile_openmp.c -lm -o sandpile_omp.exe

printf '\nChecking the exact PNG output for small and medium grid sizes ...\n'
for n in 101 201 401; do
  ./sandpile.exe "$n" >/dev/null
  cp identity.png "results/validation/ref_${n}.png"

  ./sandpile_opt.exe "$n" >/dev/null
  cmp "results/validation/ref_${n}.png" identity.png
  echo "N=$n: sequential output MATCH"

  OMP_NUM_THREADS=4 ./sandpile_omp.exe "$n" >/dev/null
  cmp "results/validation/ref_${n}.png" identity.png
  echo "N=$n: OpenMP output MATCH"
done

if [[ "${FULL:-0}" == "1" ]]; then
  printf '\nBenchmarking 1001x1001 sequential version ...\n'
  /usr/bin/time -v -o results/benchmark/sequential_1001.time ./sandpile_opt.exe 1001
  cmp results/baseline/identity_1001.png identity.png
  cp identity.png results/benchmark/sequential_1001.png
  echo 'Sequential 1001x1001 output MATCH'

  printf '\nBenchmarking 1001x1001 OpenMP version (4 threads) ...\n'
  OMP_NUM_THREADS=4 /usr/bin/time -v -o results/benchmark/openmp_1001.time ./sandpile_omp.exe 1001
  cmp results/baseline/identity_1001.png identity.png
  cp identity.png results/benchmark/openmp_1001.png
  echo 'OpenMP 1001x1001 output MATCH'

  printf '\nRESULTS:\n'
  grep -E 'Elapsed \(wall clock\)|User time|Maximum resident set size' results/benchmark/*.time
else
  printf '\nSmall/medium correctness checks passed.\n'
  printf 'To benchmark N=1001 as well, run: FULL=1 bash test_hw3.sh\n'
fi
