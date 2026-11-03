#!/usr/bin/env bash
# CS598APE HW3 - experimental multigrid + exact stabilization.
# Does not overwrite previous source files or old benchmark artifacts.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/validation results/benchmark

if [[ ! -x ./sandpile.exe ]]; then
  echo "Missing baseline ./sandpile.exe. First run: make all" >&2
  exit 1
fi
if [[ ! -f ./stb_image_write.h ]]; then
  echo "Missing stb_image_write.h in HW3 project directory" >&2
  exit 1
fi

echo "Compiling multigrid-preconditioned OpenMP (GCC -O3, 4 threads) ..."
gcc -std=c11 -O3 -g -Werror -fopenmp \
  sandpile_multigrid_openmp.c -lm -o sandpile_multigrid.exe
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE

# Includes even size (no symmetry) and three odd sizes (symmetry + multigrid).
for n in 20 101 201 401; do
  echo "Checking N=$n against original reference ..."
  ./sandpile.exe "$n" >/dev/null
  cp identity.png "results/validation/multigrid_ref_${n}.png"
  ./sandpile_multigrid.exe "$n" >/dev/null 2> "results/validation/multigrid_${n}.log"
  if ! cmp -s "results/validation/multigrid_ref_${n}.png" identity.png; then
    echo "N=$n: MISMATCH; stop, keep the existing symmetry version" >&2
    exit 1
  fi
  echo "N=$n: multigrid output MATCH"
done

echo "Small/medium correctness checks passed."
if [[ "${FULL:-0}" != "1" ]]; then
  echo "For N=1001 benchmark: FULL=1 bash test_hw3_multigrid.sh"
  exit 0
fi

reference=results/baseline/identity_1001.png
if [[ ! -f "$reference" ]]; then
  echo "Missing original baseline PNG: $reference" >&2
  echo "Not running the benchmark without a trusted N=1001 reference." >&2
  exit 1
fi

echo "Benchmarking multigrid OpenMP at N=1001 (4 threads) ..."
/usr/bin/time -v -o results/benchmark/multigrid_1001.time \
  ./sandpile_multigrid.exe 1001 \
  > results/benchmark/multigrid_1001.stdout \
  2> results/benchmark/multigrid_1001.log

if ! cmp -s "$reference" identity.png; then
  echo "N=1001: MISMATCH; do NOT use this build" >&2
  exit 1
fi
cp identity.png results/benchmark/multigrid_1001.png

echo "N=1001: multigrid output MATCH"
echo "RESULTS:"
grep -E 'User time|Elapsed \(wall clock\)|Maximum resident set size' \
  results/benchmark/multigrid_1001.time
echo "Solver diagnostics:"
cat results/benchmark/multigrid_1001.log
echo "Baseline: 611.74 seconds; previous best: 40.51 seconds."
echo "Check the elapsed time above to compute the new speedup."
