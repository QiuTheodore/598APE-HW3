#!/usr/bin/env bash
# Non-destructive V3 vs V4 comparison; V3 uses eight W-cycles; V4 uses four.
# FULL=1 RUNS=5 bash test_hw3_v4_async.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
out="$root/results/benchmark/v4_async"
mkdir -p "$out"
for f in stb_image_write.h sandpile_multigrid_v3_exchange.c sandpile_multigrid_v4_async.c;do
  [[ -f "$f" ]] || { echo "Required file missing: $f" >&2; exit 1; }
done
[[ -x "$root/sandpile.exe" ]] || { echo 'Missing sandpile.exe: build the original reference using make first.' >&2; exit 1; }
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE
flags=(-std=c11 -O3 -g -Werror -fopenmp)
echo 'Compiling V3 and V4 separately (GCC -O3, four OpenMP threads)...'
gcc "${flags[@]}" sandpile_multigrid_v3_exchange.c -lm -o "$out/v3.exe"
gcc "${flags[@]}" sandpile_multigrid_v4_async.c -lm -o "$out/v4.exe"
(
 cd "$out"
 for n in 20 101 201 401;do
   "$root/sandpile.exe" "$n" >"reference_${n}.stdout"
   mv identity.png "reference_${n}.png"
   HW3_MG_CYCLES=4 HW3_ASYNC_BUDGET=12 "$out/v4.exe" "$n" >"v4_${n}.stdout" 2>"v4_${n}.log"
   if ! cmp -s "reference_${n}.png" identity.png;then
     echo "N=$n V4 MISMATCH" >&2;exit 1
   fi
   mv identity.png "v4_${n}.png"
   echo "N=$n V4 MATCH"
 done
)
if [[ "${FULL:-0}" != 1 ]];then
  echo 'Small/medium sizes passed. Run FULL=1 RUNS=5 bash test_hw3_v4_async.sh to validate 1001 and compare performance.'
  exit 0
fi
reference="$root/results/baseline/identity_1001.png"
[[ -f "$reference" ]] || { echo "Missing original N1001 reference: $reference" >&2; exit 1; }
runs="${RUNS:-3}"
[[ "$runs" =~ ^[1-9][0-9]*$ ]] && ((runs <= 8)) || { echo 'RUNS must be an integer 1–8' >&2; exit 1; }
: >"$out/v3_times.txt";: >"$out/v4_times.txt"
run_one() {
  local which="$1" rep="$2" cycles binary
  if [[ "$which" == v3 ]];then cycles=8;binary="$out/v3.exe";else cycles=4;binary="$out/v4.exe";fi
  (
    cd "$out"
    /usr/bin/time -f '%e' -o "${which}_${rep}.seconds" \
      timeout -k 5s 75s env HW3_MG_CYCLES="$cycles" HW3_ASYNC_BUDGET=12 "$binary" 1001 \
      >"${which}_${rep}.stdout" 2>"${which}_${rep}.log"
    if ! cmp -s "$reference" identity.png;then
      echo "N=1001 $which run $rep MISMATCH" >&2;exit 1
    fi
    mv identity.png "${which}_${rep}.png"
    local fallback=""
    if grep -q 'V4 async watchdog' "${which}_${rep}.log";then fallback=' (watchdog fallback)';fi
    echo "  $which run $rep: $(cat "${which}_${rep}.seconds") s, MATCH, MG cycles=$cycles$fallback"
  )
  cat "$out/${which}_${rep}.seconds" >> "$out/${which}_times.txt"
}
echo "Cross-over N=1001 benchmark; $runs paired rounds (4 threads, guarded async V4)..."
for ((rep=1;rep<=runs;rep++));do
 if ((rep%2));then run_one v3 "$rep";run_one v4 "$rep";else run_one v4 "$rep";run_one v3 "$rep";fi
done
python3 - "$out/v3_times.txt" "$out/v4_times.txt" <<'PY'
import sys, statistics
v3=[float(s.strip()) for s in open(sys.argv[1])]
v4=[float(s.strip()) for s in open(sys.argv[2])]
for name,a in [('V3',v3),('V4',v4)]:
 print(f'{name}: mean={statistics.mean(a):.3f}s median={statistics.median(a):.3f}s best={min(a):.3f}s worst={max(a):.3f}s (n={len(a)})')
print(f'V3/V4 mean speedup={statistics.mean(v3)/statistics.mean(v4):.3f}x')
print(f'V4 vs 611.74s baseline={611.74/statistics.mean(v4):.3f}x')
print('40x target requires mean wall time below 15.2935s')
PY
echo "Comparison and exact images saved to $out"
echo 'V3 is untouched. Do NOT replace it before stable, exact VM results.'
echo 'If timeout, mismatch, or repeated fallbacks occur, reject V4 and retain V3.' 
