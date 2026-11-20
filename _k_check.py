"""Настоящая кривая Gap по k при достаточном числе итераций."""
import numpy as np
from multiprocessing import Pool
from experiment import run_config


def job(a):
    k, seed = a
    d = {r["branch"]: r for r in run_config(k, 1.5, 32, seed, T_train=500,
                                            T_test=500, n_iter=25,
                                            w_drive=0.8, nu=0.55)}
    return k, d["self"]["R"], d["external"]["R"], d["self"]["Gap"], d["self"]["conv"]


if __name__ == "__main__":
    ks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.4]
    with Pool(3) as p:
        res = p.map(job, [(k, s) for k in ks for s in range(8)])
    print(f"{'k':>5}{'R_self':>9}{'R_ext':>9}{'Gap':>10}{'sem':>9}{'conv':>8}  сошлось")
    for k in ks:
        g = np.array([[r[1], r[2], r[3], r[4]] for r in res if r[0] == k])
        conv_ok = (g[:, 3] < 0.25).mean()
        print(f"{k:5.1f}{g[:,0].mean():9.4f}{g[:,1].mean():9.4f}"
              f"{g[:,2].mean():10.4f}{g[:,2].std()/np.sqrt(len(g)):9.4f}"
              f"{np.median(g[:,3]):8.3f}  {conv_ok:.0%}")
