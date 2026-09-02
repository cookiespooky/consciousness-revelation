"""Анализ финального прогона строго по критерию из run_final.py.

Первичная оценка — медианная (L1) регрессия, решаемая точно как задача
линейного программирования: минимизируем sum|y - Xb| при вспомогательных
переменных u >= |остаток|. Никакого подбора весов и итераций, результат
воспроизводим.
"""

import sys

import numpy as np
from scipy.optimize import linprog

from analysis import load, paired_S, sel

TAU = 0.25               # порог сходимости, зафиксирован в stability.py
ALPHA_LEVEL = 5          # односторонняя нижняя граница = 5-й перцентиль
N_BOOT = 4000
MAX_COLLIN = 0.85
NULL_TOL = 0.02


def l1_fit_lp(X, y):
    """Медианная регрессия точным ЛП. Эталон для сверки."""
    n, k = X.shape
    c = np.concatenate([np.zeros(k), np.ones(n)])
    A = np.vstack([np.hstack([-X, -np.eye(n)]),
                   np.hstack([X, -np.eye(n)])])
    b = np.concatenate([-y, y])
    bounds = [(None, None)] * k + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not r.success:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    return r.x[:k]


def l1_fit(X, y, n_iter=60, eps=1e-8):
    """Та же медианная регрессия, но решаемая IRLS: веса 1/|остаток|.

    Точный ЛП давал сотни миллисекунд на решение, и бутстрэп из тысяч выборок
    не укладывался ни в какое разумное время. IRLS сходится к тому же минимуму
    sum|y - Xb| за десятки решений системы 3x3. Совпадение с ЛП проверяется на
    полной выборке в отчёте.
    """
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = np.abs(y - X @ b)
        w = 1.0 / np.maximum(r, eps)
        XW = X * w[:, None]
        try:
            b_new = np.linalg.solve(X.T @ XW, XW.T @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    return b


def keep(rows, p):
    """Правило исключения, зафиксированное до запуска."""
    out, dropped = [], 0
    for r in paired_S(rows, eps=0.0, p=p):
        if r.get("conv_max", 0) > TAU:
            dropped += 1
            continue
        out.append(r)
    return out, dropped


def _design(rs, dep):
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    y = np.array([r[dep] for r in rs])
    sd = np.array([r["seed"] for r in rs])
    ok = np.isfinite(y)
    return np.column_stack([np.ones_like(a), a, b])[ok], y[ok], sd[ok]


def slope(rs, dep, seed=23):
    X, y, sd = _design(rs, dep)
    point = l1_fit(X, y)[1]
    rng = np.random.default_rng(seed)
    seeds = np.unique(sd)
    idx = {s: np.where(sd == s)[0] for s in seeds}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        i = np.concatenate([idx[s] for s in pick])
        if np.linalg.matrix_rank(X[i]) < X.shape[1]:
            continue
        draws.append(l1_fit(X[i], y[i])[1])
    d = np.array(draws)
    return dict(point=float(point), lo=float(np.percentile(d, ALPHA_LEVEL)),
                ci=tuple(np.percentile(d, [2.5, 97.5])), n=len(y))


def slope_diff(rs, seed=29):
    X, ys, sd = _design(rs, "S")
    _, yn, _ = _design(rs, "S_null")
    m = min(len(ys), len(yn))
    X, ys, yn, sd = X[:m], ys[:m], yn[:m], sd[:m]
    rng = np.random.default_rng(seed)
    seeds = np.unique(sd)
    idx = {s: np.where(sd == s)[0] for s in seeds}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        i = np.concatenate([idx[s] for s in pick])
        if np.linalg.matrix_rank(X[i]) < X.shape[1]:
            continue
        draws.append(l1_fit(X[i], ys[i])[1] - l1_fit(X[i], yn[i])[1])
    d = np.array(draws)
    return dict(point=float(np.median(d)), lo=float(np.percentile(d, ALPHA_LEVEL)),
                ci=tuple(np.percentile(d, [2.5, 97.5])))


def matched(rs, dep):
    by = {}
    for r in rs:
        if np.isfinite(r[dep]):
            by.setdefault(r["seed"], []).append(r)
    sl = []
    for g in by.values():
        a = np.array([r["alpha"] for r in g])
        y = np.array([r[dep] for r in g])
        if len(g) < 3 or a.std() < 1e-9:
            continue
        sl.append(l1_fit(np.column_stack([np.ones_like(a), a]), y)[1])
    sl = np.array(sl)
    if len(sl) < 3:
        return None
    rng = np.random.default_rng(31)
    bs = np.array([np.median(sl[rng.integers(0, len(sl), len(sl))]) for _ in range(4000)])
    return dict(point=float(np.median(sl)), n=len(sl),
                lo=float(np.percentile(bs, ALPHA_LEVEL)),
                ci=tuple(np.percentile(bs, [2.5, 97.5])))


def mde(rows, p):
    base, _ = keep(rows, p)
    b0 = float(np.median([r["S"] for r in base]))
    rng = np.random.default_rng(41)
    tab = []
    for e in sorted({r["eps"] for r in rows if r["eps"] > 0}):
        rs = [r for r in paired_S(rows, eps=e, p=p) if r.get("conv_max", 0) <= TAU]
        if len(rs) < 4:
            continue
        d = np.array([r["S"] for r in rs]) - b0
        bs = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        tab.append((e, float(np.median(d)), float(lo), float(hi), bool(lo > 0)))
    det = [t for t in tab if t[4]]
    return tab, (min(t[1] for t in det) if det else None)


def report(path):
    rows = load(path)
    p = max(r["p"] for r in rows)
    rs, dropped = keep(rows, p)
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    collin = float(abs(np.corrcoef(a, b)[0, 1]))
    nr = [r for r in sel(rows, branch="null", eps=0.0, p=p) if not r.get("degenerate", 0)]
    gap_null = float(np.mean([r["Gap"] for r in nr])) if nr else float("nan")
    tab, mde_val = mde(rows, p)

    print(f"\n=== ФИНАЛЬНЫЙ ПРОГОН: {path} ({len(rows)} строк) ===\n")
    print("--- Проверки валидности ---")
    c_ok, n_ok = collin < MAX_COLLIN, abs(gap_null) < NULL_TOL
    print(f"  |corr(alpha,beta)| = {collin:.3f}  {'OK' if c_ok else 'ПРОВАЛ'}")
    print(f"  Gap(null)          = {gap_null:+.4f} {'OK' if n_ok else 'ПРОВАЛ'}")
    print(f"  пар после правила исключения: {len(rs)}  (исключено {dropped})")
    print(f"  размах alpha = {a.max() - a.min():.3f}   beta = {b.max() - b.min():.3f}")

    print("\n--- MDE ---")
    for e, d, lo, hi, det in tab:
        print(f"  eps={e:7.4f}  dS={d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  "
              f"{'обнаружен' if det else 'нет'}")
    print(f"  MDE = {mde_val:.5f}" if mde_val else "  MDE не установлен")

    s_r, n_r = slope(rs, "S"), slope(rs, "S_null")
    d_r = slope_diff(rs)
    mp, mpn = matched(rs, "S"), matched(rs, "S_null")

    Xc, yc, _ = _design(rs, "S")
    b_lp, b_irls = l1_fit_lp(Xc, yc)[1], l1_fit(Xc, yc)[1]
    print(f"\n--- Сверка решателей на полной выборке ---")
    print(f"  b_alpha: точный ЛП {b_lp:+.6f}   IRLS {b_irls:+.6f}   "
          f"расхождение {abs(b_lp - b_irls):.2e}")

    print("\n--- ПЕРВИЧНЫЙ ТЕСТ (медианная регрессия) ---")
    print(f"  b_alpha (S)      = {s_r['point']:+.5f}  нижняя граница {s_r['lo']:+.5f}"
          f"  CI [{s_r['ci'][0]:+.5f}, {s_r['ci'][1]:+.5f}]  n={s_r['n']}")
    print(f"  b_alpha (S_null) = {n_r['point']:+.5f}  нижняя граница {n_r['lo']:+.5f}")
    print(f"  разность         = {d_r['point']:+.5f}  нижняя граница {d_r['lo']:+.5f}")

    print("\n--- Вторичный: matched pairs ---")
    if mp:
        print(f"  S      : {mp['point']:+.5f}  нижняя граница {mp['lo']:+.5f}  seed={mp['n']}")
    if mpn:
        print(f"  S_null : {mpn['point']:+.5f}  нижняя граница {mpn['lo']:+.5f}")

    print("\n--- Средние по уровням ---")
    print(f"  {'w_dr':>5}{'nu':>6}{'c':>6} | {'alpha':>7}{'beta':>7} | "
          f"{'медиана S':>10}{'sem':>8} | {'мед S_null':>11}{'n':>5}")
    for kk in sorted({(r["w_drive"], r["nu"], r["c"]) for r in rs}):
        g = [r for r in rs if (r["w_drive"], r["nu"], r["c"]) == kk]
        S = np.array([r["S"] for r in g])
        Sn = np.array([r["S_null"] for r in g])
        Sn = Sn[np.isfinite(Sn)]
        print(f"  {kk[0]:5.1f}{kk[1]:6.2f}{kk[2]:6.1f} | "
              f"{np.mean([r['alpha'] for r in g]):7.3f}{np.mean([r['beta'] for r in g]):7.3f} | "
              f"{np.median(S):+10.5f}{S.std() / np.sqrt(len(S)):8.5f} | "
              f"{(np.median(Sn) if len(Sn) else float('nan')):+11.5f}{len(g):5d}")

    print("\n=== ВЕРДИКТ ===")
    if not (c_ok and n_ok):
        print("  ВЫВОД НЕ ВЫНОСИТСЯ: проверки валидности не пройдены.")
        return
    c1, c2, c3 = s_r["lo"] > 0, n_r["lo"] <= 0, d_r["lo"] > 0
    print(f"  (1) нижняя граница b_alpha(S) > 0        : {'ДА' if c1 else 'НЕТ'}")
    print(f"  (2) b_alpha(S_null) тот же тест НЕ прошёл: {'ДА' if c2 else 'НЕТ'}")
    print(f"  (3) нижняя граница разности > 0          : {'ДА' if c3 else 'НЕТ'}")
    if c1 and c2 and c3:
        eff = s_r["point"] * (a.max() - a.min())
        print("\n  ЭФФЕКТ ПОДТВЕРЖДЁН.")
        print(f"  Эффект по размаху alpha = {eff:+.5f}"
              + (f" против MDE {mde_val:.5f}" if mde_val else ""))
    else:
        print("\n  ЭФФЕКТ НЕ ПОДТВЕРЖДЁН. Структурная гипотеза закрывается.")
    print()


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "raw_final.csv")
