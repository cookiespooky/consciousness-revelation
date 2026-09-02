"""Запуск свипа. Пишет ТОЛЬКО сырые строки в CSV; графики и выводы — analysis.py.

  python3 run.py smoke        5 k x 2 c x 4 p x 20 seed + положительный контроль
  python3 run.py full         полный свип
  python3 run.py analyse      разбор существующего raw.csv
"""

import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

from experiment import FIELDS, run_config


def _job(args):
    """Одна конфигурация. Полностью независима: свип параллелится по ядрам."""
    k, c, p, s, e, T_tr, T_te, n_it = args
    return run_config(k, c, p, s, eps=e, T_train=T_tr, T_test=T_te, n_iter=n_it)

PROBE = dict(
    # Один вопрос: можно ли получить ненулевой S = Gap_self - Gap_yoked при
    # независимо управляемых alpha и beta. Только безопасные k (расходимость 0%),
    # широкий c для разведения alpha/beta.
    ks=[0.3, 0.55, 0.8, 1.0],
    cs=[0.0, 1.0, 2.5, 4.0],
    ps=[16, 32],
    seeds=list(range(20)),
    eps_list=[0.02, 0.05, 0.10, 0.20],
    pos_point=(0.8, 2.5),
    T_train=600, T_test=600, n_iter=6,
)

SMOKE = dict(
    ks=[0.0, 0.5, 1.0, 1.6, 2.4],
    cs=[0.0, 0.8],
    ps=[4, 8, 16, 32],
    seeds=list(range(20)),
    eps_list=[0.05, 0.15, 0.35],
    pos_point=(1.0, 0.8),            # положительный контроль в одной точке (k, c)
    T_train=600, T_test=600, n_iter=4,
)

FULL = dict(
    ks=[0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.6],
    cs=[0.0, 0.4, 0.8, 1.2],
    ps=[4, 8, 16, 32, 64, 128],
    seeds=list(range(50)),
    eps_list=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5],
    pos_point=(1.0, 0.8),
    T_train=2000, T_test=2000, n_iter=8,
)


def sweep(cfg, out_path):
    jobs = list(itertools.product(cfg["ks"], cfg["cs"], cfg["ps"], cfg["seeds"]))
    pmax = max(cfg["ps"])
    pk, pc = cfg["pos_point"]
    pos = [(pk, pc, pmax, s, e)
           for s, e in itertools.product(cfg["seeds"], cfg["eps_list"])]
    total = len(jobs) + len(pos)
    print(f"конфигураций: {len(jobs)} основных + {len(pos)} положительного "
          f"контроля = {total};  4 ветки на каждую")

    args = ([(k, c, p, s, 0.0, cfg["T_train"], cfg["T_test"], cfg["n_iter"])
             for k, c, p, s in jobs]
            + [(k, c, p, s, e, cfg["T_train"], cfg["T_test"], cfg["n_iter"])
               for k, c, p, s, e in pos])

    t0 = time.time()
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        with Pool(n_proc) as pool:
            for i, rows in enumerate(pool.imap_unordered(_job, args, chunksize=4), 1):
                for row in rows:
                    w.writerow({q: row[q] for q in FIELDS})
                if i % 40 == 0:
                    el = time.time() - t0
                    print(f"  {i}/{total}  {el:6.1f}s  ~{el / i * (total - i):6.1f}s осталось",
                          flush=True)
                    f.flush()
    print(f"готово за {time.time() - t0:.1f}s на {n_proc} процессах -> {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    out = sys.argv[2] if len(sys.argv) > 2 else f"raw_{mode}.csv"
    if mode == "analyse":
        from analysis import report
        report(out)
    else:
        sweep({"probe": PROBE, "smoke": SMOKE, "full": FULL}[mode], out)
        from analysis import report
        report(out)
