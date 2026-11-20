"""Мелкая сетка вокруг порога: где он и одинаков ли он по seed.

Порог определяется НЕ по Gap (это было бы кругом), а по сходимости попеременной
подгонки — критерий conv < 0.25, введённый ранее и здесь не подбиравшийся.
"""
import numpy as np
from multiprocessing import Pool
from experiment import run_config

KS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
SEEDS = list(range(16))
TAU = 0.25


def job(a):
    k, seed = a
    d = {r["branch"]: r for r in run_config(k, 1.5, 32, seed, T_train=500,
                                            T_test=500, n_iter=20,
                                            w_drive=0.8, nu=0.55)}
    return k, seed, d["self"]["R"], d["external"]["R"], d["self"]["Gap"], d["self"]["conv"]


if __name__ == "__main__":
    with Pool(3) as p:
        res = p.map(job, [(k, s) for k in KS for s in SEEDS])

    print(f"{'k':>6}{'мед Gap':>10}{'сред Gap':>10}{'R_ext':>9}{'сошлось':>9}")
    for k in KS:
        g = np.array([[r[2], r[3], r[4], r[5]] for r in res if r[0] == k])
        print(f"{k:6.2f}{np.median(g[:,2]):10.4f}{g[:,2].mean():10.4f}"
              f"{g[:,1].mean():9.4f}{(g[:,3] < TAU).mean():9.0%}")

    # порог на seed: наибольшее k, до которого сходимость держалась непрерывно
    thr = []
    for s in SEEDS:
        run = [r for r in sorted(res) if r[1] == s]
        last = np.nan
        for r in run:
            if r[5] < TAU:
                last = r[0]
            else:
                break
        thr.append(last)
    thr = np.array(thr, dtype=float)
    ok = np.isfinite(thr)
    print(f"\nпороги по seed: {sorted(thr[ok])}")
    print(f"медиана {np.median(thr[ok]):.2f}, разброс {thr[ok].min():.2f}..{thr[ok].max():.2f}, "
          f"std {thr[ok].std():.3f}, определён у {ok.sum()}/{len(SEEDS)}")
