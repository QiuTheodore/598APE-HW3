#!/usr/bin/env bash
# Non-destructive profiling of the verified HW3 multigrid V2 implementation.
# Usage: bash profile_hw3_v2.sh
#        FULL=1 bash profile_hw3_v2.sh
#        FULL=1 RUN_PERF=1 bash profile_hw3_v2.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
: "${OMP_NUM_THREADS:=4}"
export OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export HW3_MG_CYCLES=8
out="$root/results/profile/v2"
mkdir -p "$out"
if [[ ! -f "$root/stb_image_write.h" ]]; then
  echo "Missing stb_image_write.h in $root" >&2; exit 1
fi
if [[ ! -x "$root/sandpile.exe" ]]; then
  echo 'Missing sandpile.exe; run make all first.' >&2; exit 1
fi
printf 'Compiling independent profiling builds (GCC -O3)...\n'
gcc -std=c11 -O3 -g -Werror -fopenmp \
  "$root/sandpile_profile_v2.c" -lm -o "$out/sandpile_profile_v2.exe"
gcc -std=c11 -O3 -g -Werror -fopenmp \
  "$root/sandpile_profile_threads.c" -lm -o "$out/sandpile_profile_threads.exe"
printf 'Correctness tests with the original reference (in isolated profile directory)...\n'
(
  cd "$out"
  for n in 20 101 201 401; do
    "$root/sandpile.exe" "$n" >/dev/null
    mv identity.png "ref_${n}.png"
    "$out/sandpile_profile_v2.exe" "$n" > "profile_${n}.stdout" 2> "profile_${n}.log"
    if ! cmp -s "ref_${n}.png" identity.png; then
      echo "N=$n: MISMATCH; profiling build is invalid." >&2; exit 1
    fi
    echo "N=$n: MATCH"
  done
)
if [[ "${FULL:-0}" != 1 ]]; then
  printf 'Small/medium tests passed. To profile N=1001, run: FULL=1 bash profile_hw3_v2.sh\n'
  exit 0
fi
ref="$root/results/baseline/identity_1001.png"
if [[ ! -f "$ref" ]]; then
  echo "Missing reference PNG: $ref" >&2; exit 1
fi
printf 'Profiling N=1001 (4 threads) with isolated outputs...\n'
(
  cd "$out"
  /usr/bin/time -v -o profile_1001.time \
    "$out/sandpile_profile_v2.exe" 1001 \
    > profile_1001.stdout 2> profile_1001.log
  if ! cmp -s "$ref" identity.png; then
    echo 'N=1001: MISMATCH; do not use this result.' >&2; exit 1
  fi
  mv identity.png profile_1001.png
  echo 'N=1001: MATCH'
  /usr/bin/time -v -o threads_1001.time \
    "$out/sandpile_profile_threads.exe" 1001 \
    > threads_1001.stdout 2> threads_1001.log
  if ! cmp -s "$ref" identity.png; then
    echo 'N=1001 thread-profile: MISMATCH' >&2; exit 1
  fi
  mv identity.png threads_1001.png
)
printf '\nStage breakdown:\n'
grep -E '^PROFILE |^MG cycles' "$out/profile_1001.log"
printf '\nThread workload and wait breakdown:\n'
grep -E '^PROFILE thread ' "$out/threads_1001.log"
printf '\nInstrumented elapsed time (not official benchmark):\n'
grep -E 'Elapsed \(wall clock\)' "$out/profile_1001.time" "$out/threads_1001.time"
if [[ "${RUN_PERF:-0}" == 1 ]]; then
  if ! command -v perf >/dev/null 2>&1; then
    echo 'perf not found on this VM. Stage/thread profiles were still saved.'
  else
    echo 'Trying perf record on N=1001. This is a separate profiling run.'
    (
      cd "$out"
      if perf record -q -F 99 -g -o perf.data -- \
          "$out/sandpile_profile_v2.exe" 1001 > perf.stdout 2> perf.stderr; then
        perf report --stdio -i perf.data --no-children > perf_report.txt 2> perf_report.stderr || true
        echo "perf report: $out/perf_report.txt"
      else
        echo 'perf sampling unavailable (possibly perf_event_paranoid); see perf.stderr.'
      fi
    )
  fi
fi
printf '\nSaved reports under: %s\n' "$out"
