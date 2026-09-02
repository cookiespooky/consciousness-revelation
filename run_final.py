"""ФИНАЛЬНЫЙ прогон структурной ветки. Всё зафиксировано до запуска.

ПРЕДРЕГИСТРАЦИЯ
---------------
Первичная оценка:  МЕДИАННАЯ (L1) регрессия  S = b0 + b_a*alpha + b_b*beta.
                   Выбрана заранее и именно потому, что обычная МНК-оценка
                   оказалась чувствительна к редким численным срывам; винзоризация
                   отвергнута в пользу медианной регрессии.
Гипотезы:          H0: b_alpha <= 0   против   H1: b_alpha > 0.
Критерий (все три обязательны):
  (1) односторонняя нижняя граница 95% (5-й перцентиль кластерного бутстрэпа
      по seed) для b_alpha(S) строго больше нуля;
  (2) тот же критерий для b_alpha(S_null) НЕ выполняется;
  (3) нижняя граница разности наклонов S - S_null строго больше нуля.
Правило исключения (ЗАФИКСИРОВАНО ДО ЗАПУСКА, после просмотра не меняется):
  конфигурация исключается, если max(conv_self, conv_yoked) превышает
  TAU = 0.25 — порог сходимости попеременной подгонки, введённый ранее в
  stability.py и здесь не подбиравшийся; либо если помечено вырождение цели.
  (Отдельная conv ветки self2 в CSV не пишется; диагностика прошлого прогона
  показала, что conv у self и self2 движутся вместе — медианы совпали, 0.273
  против 0.273 на срывах и 0.056 против 0.056 на нормальных конфигурациях.)
Вторичный эндпоинт: matched pairs — наклон по alpha ВНУТРИ каждого seed,
  затем усреднение. Не заменяет первичный тест ни при каком исходе.
Валидность: |corr(alpha,beta)| < 0.85; |Gap(null)| < 0.02; MDE установлен
  мелкой сеткой eps.
Метрики k_c, lambda_max, rho* в этом прогоне НЕ анализируются.

Конструкция
-----------
k = 0.4 фиксировано во всех точках: меняется только доля экзогенной причинности
тела (w_drive, nu). Иначе k вошёл бы в модель пропущенной переменной.
w_drive >= 0.5 везде: срывы прошлого прогона все до одного случались при
w_drive = 0.4, где медиана conv была 0.273 против 0.056 у остальных.
seed: 30 вместо 20 — компенсация за сузившийся размах alpha (0.145..0.501).
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
LEVELS = [(1.5, 0.55), (1.5, 0.25), (0.8, 0.55), (0.8, 0.05), (0.6, 0.05), (0.5, 0.05)]
P_CAP = 32
SEEDS = list(range(30))
EPS_LIST = [0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
POS_LEVEL, POS_C = (0.8, 0.55), 1.5
T_TRAIN = T_TEST = 500
N_ITER = 6


def job(a):
    w, nu, c, s, e = a
    return run_config(K, c, P_CAP, s, eps=e, T_train=T_TRAIN, T_test=T_TEST,
                      n_iter=N_ITER, w_drive=w, nu=nu)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "raw_final.csv"
    args = [(w, nu, c, s, 0.0)
            for (w, nu), c, s in itertools.product(LEVELS, CS, SEEDS)]
    args += [(POS_LEVEL[0], POS_LEVEL[1], POS_C, s, e)
             for s, e in itertools.product(SEEDS, EPS_LIST)]
    t0 = time.time()
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    print(f"конфигураций: {len(args)}; {n_proc} процессов", flush=True)
    with open(out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=FIELDS)
        w_.writeheader()
        with Pool(n_proc) as pool:
            for i, rows in enumerate(pool.imap_unordered(job, args, chunksize=4), 1):
                for r in rows:
                    w_.writerow({q: r[q] for q in FIELDS})
                if i % 60 == 0:
                    el = time.time() - t0
                    f.flush()
                    print(f"  {i}/{len(args)} {el:6.1f}s "
                          f"~{el / i * (len(args) - i):6.1f}s осталось", flush=True)
    print(f"готово за {time.time() - t0:.1f}s -> {out}")
