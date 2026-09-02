"""Прогон границы устойчивости: k_c(self) против k_c(yoked) по seed."""
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from stability import FIELDS, run_pair

K_GRID = np.arange(0.0, 1.61, 0.2)
SEEDS = list(range(12))
CS = [1.0]
# Та же лестница по alpha, что в run_alpha.py: меняется доля экзогенной
# причинности тела, k при этом сканируется для поиска порога.
LEVELS = [(1.5, 0.25), (0.4, 0.55), (0.4, 0.05), (0.1, 0.05)]
P_CAP = 32
T_TRAIN, N_ITER = 500, 6

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "raw_stability.csv"
    contrast = sys.argv[2] if len(sys.argv) > 2 else "yoked"
    args = [(s, c, P_CAP, K_GRID, T_TRAIN, N_ITER, contrast, w, nu)
            for s, c, (w, nu) in itertools.product(SEEDS, CS, LEVELS)]
    t0 = time.time()
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    print(f"контраст: self vs {contrast};  пар (seed, c): {len(args)};  "
          f"сетка k: {len(K_GRID)} точек; {n_proc} процессов", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        with Pool(n_proc) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_pair, args), 1):
                for r in rows:
                    w.writerow({q: r[q] for q in FIELDS})
                f.flush()
                el = time.time() - t0
                print(f"  {i}/{len(args)}  {el:6.1f}s  "
                      f"~{el / i * (len(args) - i):6.1f}s осталось", flush=True)
    print(f"готово за {time.time() - t0:.1f}s -> {out}")
