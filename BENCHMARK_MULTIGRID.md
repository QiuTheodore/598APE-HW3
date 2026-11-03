# CS598APE HW3 - Best Validated Version

Date: 2026-11-03
Machine: UIUC course VM, 4 vCPUs
Workload: N=1001
Algorithm: Symmetry + Multigrid W-cycle + OpenMP

Baseline: 611.74 s
Run 1: 19.50 s
Run 2: 19.49 s
Best speedup: 31.39x

Correctness: PNG MATCH at N=20,101,201,401,1001
Peak memory (run 2): 39156 KB

Source: sandpile_multigrid_openmp.c
Test: test_hw3_multigrid.sh
Command: FULL=1 bash test_hw3_multigrid.sh
