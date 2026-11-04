#!/usr/bin/env bash
# CS598APE HW3: independent correctness validation and v2 benchmark.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/validation results/benchmark
if [[ ! -x ./sandpile.exe ]]; then echo 'Missing ./sandpile.exe; run make all first' >&2; exit 1; fi
if [[ ! -f ./stb_image_write.h ]]; then echo 'Missing stb_image_write.h' >&2; exit 1; fi

echo 'Compiling multigrid v2 with GCC -O3 (4 OpenMP threads) ...'
gcc -std=c11 -O3 -g -Werror -fopenmp sandpile_multigrid_v2.c -lm -o sandpile_multigrid_v2.exe
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE HW3_MG_CYCLES=8

for n in 20 101 201 401; do
  echo "Checking N=$n against the original reference ..."
  ./sandpile.exe "$n" >/dev/null
  cp identity.png "results/validation/multigrid_v2_ref_${n}.png"
  ./sandpile_multigrid_v2.exe "$n" > /dev/null 2> "results/validation/multigrid_v2_${n}.log"
  if ! cmp -s "results/validation/multigrid_v2_ref_${n}.png" identity.png; then
    echo "N=$n: MISMATCH; do not use v2" >&2; exit 1
  fi
  echo "N=$n: multigrid v2 output MATCH"
done

echo 'Small/medium correctness checks passed.'
if [[ "${FULL:-0}" != 1 ]]; then
  echo 'For N=1001: FULL=1 bash test_hw3_multigrid_v2.sh'
  exit 0
fi

reference=results/baseline/identity_1001.png
if [[ ! -f "$reference" ]]; then echo "Missing original baseline PNG $reference" >&2; exit 1; fi

echo 'Benchmarking multigrid v2 at N=1001 (4 threads) ...'
/usr/bin/time -v -o results/benchmark/multigrid_v2_1001.time \
  ./sandpile_multigrid_v2.exe 1001 \
  > results/benchmark/multigrid_v2_1001.stdout \
  2> results/benchmark/multigrid_v2_1001.log
if ! cmp -s "$reference" identity.png; then
  echo 'N=1001: MISMATCH; do not use v2' >&2; exit 1
fi
cp identity.png results/benchmark/multigrid_v2_1001.png
echo 'N=1001: multigrid v2 output MATCH'
echo 'RESULTS:'
grep -E 'User time|Elapsed \(wall clock\)|Maximum resident set size' results/benchmark/multigrid_v2_1001.time
echo 'Solver diagnostics:'
cat results/benchmark/multigrid_v2_1001.log
echo 'Saved best: 19.49 s; for 40x aim under 15.29 s.'
