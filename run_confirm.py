"""Подтверждающий прогон. Одна гипотеза, один первичный эндпоинт.

ПРЕДРЕГИСТРАЦИЯ (зафиксировано до запуска, менять после просмотра данных нельзя)
------------------------------------------------------------------------------
Первичная модель:      S = b0 + b_alpha * alpha + b_beta * beta + e
                       на self/yoked-парах, p = 32, eps = 0, невырожденные.
Гипотезы:              H0: b_alpha <= 0   против   H1: b_alpha > 0.
Критерий обнаружения (оба условия обязательны):
    (1) односторонняя нижняя граница 95% (5-й перцентиль кластерного
        бутстрэпа по seed) для b_alpha строго больше нуля;
    (2) тот же наклон, посчитанный для S_null вместо S, критерий (1) НЕ проходит,
        и разность наклонов S - S_null имеет нижнюю границу выше нуля.
Дополнительно (не заменяет первичный тест):
    matched pairs — наклон по alpha считается ВНУТРИ каждого seed, затем
    усредняется. Убирает разброс между случайными базисами целиком.
Валидность прогона требует:
    - |corr(alpha, beta)| < 0.85, иначе b_alpha неидентифицируем;
    - Gap(null) в пределах +-0.02, иначе установка не воспроизводит ноль;
    - MDE, установленный мелкой сеткой eps, а не её нижней границей.

Что изменено против разведочного прогона и зачем
------------------------------------------------
c        широкий  {0, 1.5, 4.0}: развести alpha и beta, в лестнице они шли
                  вместе (corr 0.883) и b_alpha был неидентифицируем.
w_drive  >= 0.4:  при слабом экзогенном приводе у null-ветки (k=0) телу нечем
                  двигаться, цель схлопывается и Gap(null) уходил в -0.20.
                  Заодно исчезает отбраковка по вырождению (была 50-65% на
                  верхних уровнях), то есть уходит и селекция.
eps      мелкая сетка 0.005..0.03: прежний MDE = 0.024 был наименьшим
                  ПРОТЕСТИРОВАННЫМ значением, то есть верхней границей.
seed     тот же набор.
"""
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

from experiment import FIELDS, run_config

K = 0.4
CS = [0.0, 1.5, 4.0]
LEVELS = [(1.5, 0.55), (1.5, 0.25), (0.8, 0.55), (0.4, 0.55), (0.4, 0.25), (0.4, 0.05)]
P_CAP = 32
SEEDS = list(range(20))
EPS_LIST = [0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
POS_LEVEL = (0.8, 0.55)
POS_C = 1.5
T_TRAIN = T_TEST = 500
N_ITER = 6


def job(a):
    w, nu, c, s, e = a
    return run_config(K, c, P_CAP, s, eps=e, T_train=T_TRAIN, T_test=T_TEST,
                      n_iter=N_ITER, w_drive=w, nu=nu)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "raw_confirm.csv"
    args = [(w, nu, c, s, 0.0)
            for (w, nu), c, s in itertools.product(LEVELS, CS, SEEDS)]
    args += [(POS_LEVEL[0], POS_LEVEL[1], POS_C, s, e)
             for s, e in itertools.product(SEEDS, EPS_LIST)]
    t0 = time.time()
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    print(f"конфигураций: {len(args)} ({len(LEVELS)} уровней x {len(CS)} c x "
          f"{len(SEEDS)} seed + {len(EPS_LIST) * len(SEEDS)} калибровочных); "
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
