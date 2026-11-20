"""Пропущенный контроль: не схлопывается ли разрыв при росте числа итераций.

Если Gap = R_self - R_external падает к нулю с ростом n_iter, значит мы мерили
недосходимость подгонки, а не структурный предел.
"""
import numpy as np
from multiprocessing import Pool
from experiment import run_config


def job(a):
    ni, seed = a
    d = {r["branch"]: r for r in run_config(0.4, 1.5, 32, seed, T_train=500,
                                            T_test=500, n_iter=ni,
                                            w_drive=0.8, nu=0.55)}
    return ni, d["self"]["R"], d["external"]["R"], d["self"]["Gap"], d["self"]["conv"]


if __name__ == "__main__":
    args = [(ni, s) for ni in [4, 6, 10, 20, 40] for s in range(8)]
    with Pool(3) as p:
        res = p.map(job, args)
    print(f"{'n_iter':>7}{'R_self':>9}{'R_ext':>9}{'Gap':>10}{'sem':>9}{'conv':>8}")
    for ni in [4, 6, 10, 20, 40]:
        g = np.array([[r[1], r[2], r[3], r[4]] for r in res if r[0] == ni])
        print(f"{ni:7d}{g[:,0].mean():9.4f}{g[:,1].mean():9.4f}"
              f"{g[:,2].mean():10.5f}{g[:,2].std()/np.sqrt(len(g)):9.5f}{g[:,3].mean():8.3f}")
