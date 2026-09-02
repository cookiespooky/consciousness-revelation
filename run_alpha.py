"""Лестница по alpha при фиксированном k: 0.19 -> 0.90.

k держится равным 0.4 во всех точках. Меняется только доля экзогенной
причинности тела (w_drive, nu), то есть ровно то, насколько система моделирует
собственную причинную динамику, а не кусок тела, ведомый извне.
"""
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

from experiment import FIELDS, run_config

K = 0.4
C = 1.0
# w_drive нигде не ноль: при полном отсутствии экзогенного привода тело теряет
# вариативность, цель схлопывается до шума и метрики теряют смысл. alpha ~ 0.9
# достижима только в этом вырожденном пределе, поэтому верх лестницы ~0.78.
LEVELS = [(1.5, 0.25), (0.8, 0.55), (0.4, 0.55), (0.4, 0.05), (0.1, 0.25), (0.1, 0.05)]
PS = [8, 16, 32]
SEEDS = list(range(20))
EPS_LIST = [0.02, 0.05, 0.10, 0.20]
POS_LEVEL = (0.4, 0.55)
T_TRAIN = T_TEST = 600
N_ITER = 6


def job(a):
    w, nu, p, s, e = a
    return run_config(K, C, p, s, eps=e, T_train=T_TRAIN, T_test=T_TEST,
                      n_iter=N_ITER, w_drive=w, nu=nu)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "raw_alpha.csv"
    pmax = max(PS)
    args = [(w, nu, p, s, 0.0)
            for (w, nu), p, s in itertools.product(LEVELS, PS, SEEDS)]
    args += [(POS_LEVEL[0], POS_LEVEL[1], pmax, s, e)
             for s, e in itertools.product(SEEDS, EPS_LIST)]
    t0 = time.time()
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    print(f"конфигураций: {len(args)}; уровней alpha: {len(LEVELS)}; "
          f"{n_proc} процессов", flush=True)
    with open(out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=FIELDS)
        w_.writeheader()
        with Pool(n_proc) as pool:
            for i, rows in enumerate(pool.imap_unordered(job, args, chunksize=4), 1):
                for r in rows:
                    w_.writerow({q: r[q] for q in FIELDS})
                if i % 40 == 0:
                    el = time.time() - t0
                    f.flush()
                    print(f"  {i}/{len(args)} {el:6.1f}s "
                          f"~{el / i * (len(args) - i):6.1f}s осталось", flush=True)
    print(f"готово за {time.time() - t0:.1f}s -> {out}")
