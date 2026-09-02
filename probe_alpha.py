"""Калибровка ручек: какие (k, w_drive, nu) дают нужные уровни alpha."""
import itertools
from multiprocessing import Pool

import numpy as np

from dynamics import Params
from experiment import measure_alpha_beta
from model import Basis, fit_closed_loop


def job(a):
    k, w, nu = a
    out = []
    for seed in range(3):
        par = Params(k=k, c=1.0, seed=seed, w_drive=w, nu=nu)
        basis = Basis(16, seed)
        G, _, hist = fit_closed_loop(par, basis, "self", 500, 6)
        al, be = measure_alpha_beta(par, G, None, "self")
        out.append((al, be, float(np.median(hist[-3:]))))
    A = np.array(out)
    return (k, w, nu, A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean())


if __name__ == "__main__":
    grid = list(itertools.product([0.3, 0.4, 0.6, 0.8],
                                  [1.5, 0.8, 0.4, 0.15, 0.0],
                                  [0.55, 0.25, 0.05]))
    with Pool(3) as p:
        res = p.map(job, grid)
    res.sort(key=lambda r: r[3])
    print(f"{'k':>5}{'w_dr':>6}{'nu':>6} | {'alpha':>7}{'beta':>7}{'conv':>7}")
    for k, w, nu, al, be, cv in res:
        flag = "  <-" if cv < 0.25 else "   расх."
        print(f"{k:5.1f}{w:6.1f}{nu:6.2f} | {al:7.3f}{be:7.3f}{cv:7.3f}{flag}")
