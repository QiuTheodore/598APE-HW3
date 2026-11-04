#!/usr/bin/env bash
# CS598APE HW3: non-destructive V2 vs V3 bounded-exchange A/B benchmark.
# Usage: bash test_hw3_v3_exchange.sh
#        FULL=1 bash test_hw3_v3_exchange.sh
#        FULL=1 RUNS=2 bash test_hw3_v3_exchange.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
out="$root/results/benchmark/v3_exchange"
mkdir -p "$out"
for file in stb_image_write.h sandpile_multigrid_v2.c sandpile_multigrid_v3_exchange.c; do
  if [[ ! -f "$root/$file" ]]; then
    echo "Missing required file: $root/$file" >&2
    exit 1
  fi
done
if [[ ! -x "$root/sandpile.exe" ]]; then
  echo 'Missing original sandpile.exe; run make all first.' >&2
  exit 1
fi
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE HW3_MG_CYCLES=8
# Preserve V2 source, V2 binary, and all benchmark artifacts outside $out.
flags=(-std=c11 -O3 -g -Werror -fopenmp)
echo 'Building independent V2/V3 comparison binaries using the same flags...'
gcc "${flags[@]}" "$root/sandpile_multigrid_v2.c" -lm -o "$out/v2.exe"
gcc "${flags[@]}" "$root/sandpile_multigrid_v3_exchange.c" -lm -o "$out/v3_exchange.exe"

(
  cd "$out"
  echo 'Verifying V3 against original reference on N=20,101,201,401...'
  for n in 20 101 201 401; do
    "$root/sandpile.exe" "$n" >"ref_${n}.stdout"
    mv identity.png "reference_${n}.png"
    "$out/v3_exchange.exe" "$n" >"v3_${n}.stdout" 2>"v3_${n}.log"
    if ! cmp -s "reference_${n}.png" identity.png; then
      echo "N=$n MISMATCH. Do not use V3." >&2
      exit 1
    fi
    mv identity.png "v3_${n}.png"
    echo "N=$n MATCH"
  done
)
if [[ "${FULL:-0}" != 1 ]]; then
  echo 'Small/medium checks passed. For N=1001 comparison: FULL=1 bash test_hw3_v3_exchange.sh'
  exit 0
fi
reference="$root/results/baseline/identity_1001.png"
if [[ ! -f "$reference" ]]; then
  echo "Missing original N=1001 reference: $reference" >&2
  exit 1
fi
runs="${RUNS:-2}"
if ! [[ "$runs" =~ ^[1-9][0-9]*$ ]] || ((runs > 8)); then
  echo 'RUNS must be an integer between 1 and 8.' >&2
  exit 1
fi
printf 'N=1001 V2/V3 comparison, %s rounds, 4 OpenMP threads.\n' "$runs"
: > "$out/v2_times.txt"
: > "$out/v3_times.txt"
run_one() {
  local version="$1" rep="$2" binary
  if [[ "$version" == v2 ]]; then binary="$out/v2.exe"; else binary="$out/v3_exchange.exe"; fi
  (
    cd "$out"
    # Kill unexpectedly slow experiments instead of leaving the VM stuck.
    /usr/bin/time -f '%e' -o "${version}_${rep}.seconds" \
      timeout -k 5s 75s "$binary" 1001 \
      >"${version}_${rep}.stdout" 2>"${version}_${rep}.log"
    if ! cmp -s "$reference" identity.png; then
      echo "N=1001 $version run $rep MISMATCH" >&2
      exit 1
    fi
    mv identity.png "${version}_${rep}.png"
    echo "  $version run $rep: $(cat "${version}_${rep}.seconds") seconds, MATCH"
  )
  cat "$out/${version}_${rep}.seconds" >> "$out/${version}_times.txt"
}
for ((rep=1;rep<=runs;rep++)); do
  if ((rep%2 == 1)); then
    run_one v2 "$rep"
    run_one v3 "$rep"
  else
    run_one v3 "$rep"
    run_one v2 "$rep"
  fi
done
for v in v2 v3; do
  awk -v name="$v" '{sum+=$1; if(min==0 || $1<min)min=$1; if($1>max)max=$1; n++}END{printf "%s: mean=%.3f s  best=%.3f s  worst=%.3f s  runs=%d\n",name,sum/n,min,max,n}' "$out/${v}_times.txt"
done
awk 'NR==FNR{a+=$1;n++;next}{b+=$1;m++}END{printf "V2/V3 mean ratio=%.3fx (>1 means V3 faster)\n",(a/n)/(b/m)}' "$out/v2_times.txt" "$out/v3_times.txt"
echo 'This is an experimental candidate. Do NOT replace your validated V2 automatically.'
echo "Saved independent output under: $out"
