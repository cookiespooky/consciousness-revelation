"""Анализ подтверждающего прогона по критерию, зафиксированному в run_confirm.py.

Порядок жёсткий: сначала проверки валидности, и только если они прошли —
первичный тест. Если предпосылки нарушены, вердикт не выносится вообще.
"""

import sys

import numpy as np

from analysis import load, paired_S, sel

ALPHA_LEVEL = 5          # односторонняя нижняя граница = 5-й перцентиль
N_BOOT = 8000
MAX_COLLIN = 0.85
NULL_TOL = 0.02


def _design(rs, dep):
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    y = np.array([r[dep] for r in rs])
    return np.column_stack([np.ones_like(a), a, b]), y, np.array([r["seed"] for r in rs])


def slope_ci(rs, dep="S", n_boot=N_BOOT, seed=23):
    """Наклон по alpha при beta в ковариатах. Кластерный бутстрэп по seed."""
    X, y, sd = _design(rs, dep)
    ok = np.isfinite(y)
    X, y, sd = X[ok], y[ok], sd[ok]
    point = np.linalg.lstsq(X, y, rcond=None)[0][1]
    rng = np.random.default_rng(seed)
    seeds = np.unique(sd)
    idx_by = {s: np.where(sd == s)[0] for s in seeds}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, len(seeds), replace=True)
        i = np.concatenate([idx_by[s] for s in pick])
        if np.linalg.matrix_rank(X[i]) < X.shape[1]:
            continue
        out.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    out = np.array(out)
    return dict(point=float(point), one_sided_lo=float(np.percentile(out, ALPHA_LEVEL)),
                ci=tuple(np.percentile(out, [2.5, 97.5])), draws=out, n=int(ok.sum()))


def slope_diff(rs, n_boot=N_BOOT, seed=29):
    """Разность наклонов S и S_null на одних и тех же бутстрэп-выборках."""
    Xs, ys, sd = _design(rs, "S")
    _, yn, _ = _design(rs, "S_null")
    ok = np.isfinite(ys) & np.isfinite(yn)
    Xs, ys, yn, sd = Xs[ok], ys[ok], yn[ok], sd[ok]
    rng = np.random.default_rng(seed)
    seeds = np.unique(sd)
    idx_by = {s: np.where(sd == s)[0] for s in seeds}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, len(seeds), replace=True)
        i = np.concatenate([idx_by[s] for s in pick])
        if np.linalg.matrix_rank(Xs[i]) < Xs.shape[1]:
            continue
        bs = np.linalg.lstsq(Xs[i], ys[i], rcond=None)[0][1]
        bn = np.linalg.lstsq(Xs[i], yn[i], rcond=None)[0][1]
        out.append(bs - bn)
    out = np.array(out)
    return dict(point=float(np.mean(out)), one_sided_lo=float(np.percentile(out, ALPHA_LEVEL)),
                ci=tuple(np.percentile(out, [2.5, 97.5])))


def matched_pairs(rs, dep="S"):
    """Наклон по alpha ВНУТРИ каждого seed, затем усреднение по seed.

    Разброс между случайными базисами уже показал, что способен маскировать
    эффект; внутрисидовая оценка убирает его целиком.
    """
    by = {}
    for r in rs:
        if np.isfinite(r[dep]):
            by.setdefault(r["seed"], []).append(r)
    slopes = []
    for s, g in by.items():
        a = np.array([r["alpha"] for r in g])
        y = np.array([r[dep] for r in g])
        if len(g) < 3 or a.std() < 1e-9:
            continue
        slopes.append(np.polyfit(a, y, 1)[0])
    slopes = np.array(slopes)
    if len(slopes) < 3:
        return None
    rng = np.random.default_rng(31)
    bs = np.array([slopes[rng.integers(0, len(slopes), len(slopes))].mean()
                   for _ in range(N_BOOT)])
    return dict(point=float(slopes.mean()), n_seeds=len(slopes),
                one_sided_lo=float(np.percentile(bs, ALPHA_LEVEL)),
                ci=tuple(np.percentile(bs, [2.5, 97.5])))


def mde(rows, p):
    base = [r for r in paired_S(rows, eps=0.0, p=p)]
    b0 = float(np.mean([r["S"] for r in base]))
    rng = np.random.default_rng(41)
    tab = []
    for e in sorted({r["eps"] for r in rows if r["eps"] > 0}):
        rs = paired_S(rows, eps=e, p=p)
        if len(rs) < 4:
            continue
        d = np.array([r["S"] for r in rs]) - b0
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        tab.append((e, float(d.mean()), float(lo), float(hi), bool(lo > 0)))
    det = [t for t in tab if t[4]]
    return tab, (min(t[1] for t in det) if det else None)


def report(path):
    rows = load(path)
    p = max(r["p"] for r in rows)
    rs = paired_S(rows, eps=0.0, p=p)
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    collin = float(abs(np.corrcoef(a, b)[0, 1]))
    nullrows = [r for r in sel(rows, branch="null", eps=0.0, p=p)
                if not r.get("degenerate", 0)]
    gap_null = float(np.mean([r["Gap"] for r in nullrows])) if nullrows else float("nan")
    tab, mde_val = mde(rows, p)

    print(f"\n=== ПОДТВЕРЖДАЮЩИЙ ПРОГОН: {path} ({len(rows)} строк) ===\n")
    print("--- Проверки валидности (до первичного теста) ---")
    c_ok = collin < MAX_COLLIN
    n_ok = abs(gap_null) < NULL_TOL
    print(f"  |corr(alpha,beta)| = {collin:.3f}   {'OK' if c_ok else 'ПРОВАЛ'} "
          f"(нужно < {MAX_COLLIN})")
    print(f"  Gap(null)          = {gap_null:+.4f}  {'OK' if n_ok else 'ПРОВАЛ'} "
          f"(нужно |.| < {NULL_TOL})")
    print(f"  размах alpha       = {a.max() - a.min():.3f}   размах beta = {b.max() - b.min():.3f}")
    print(f"  пар n              = {len(rs)}")

    print("\n--- MDE (мелкая сетка eps) ---")
    print(f"  {'eps':>7} {'dS':>9} {'CI низ':>9} {'CI верх':>9}  обнаружен")
    for e, d, lo, hi, det in tab:
        print(f"  {e:7.4f} {d:+9.5f} {lo:+9.5f} {hi:+9.5f}  {'да' if det else 'нет'}")
    print(f"  MDE = {mde_val:.5f}" if mde_val else "  MDE не установлен")

    s_res = slope_ci(rs, "S")
    n_res = slope_ci(rs, "S_null")
    d_res = slope_diff(rs)
    mp = matched_pairs(rs, "S")
    mpn = matched_pairs(rs, "S_null")

    print("\n--- ПЕРВИЧНЫЙ ТЕСТ: S ~ 1 + alpha + beta ---")
    print(f"  b_alpha (S)      = {s_res['point']:+.5f}   односторонняя нижняя "
          f"граница = {s_res['one_sided_lo']:+.5f}   двусторонний CI "
          f"[{s_res['ci'][0]:+.5f}, {s_res['ci'][1]:+.5f}]")
    print(f"  b_alpha (S_null) = {n_res['point']:+.5f}   нижняя граница = "
          f"{n_res['one_sided_lo']:+.5f}")
    print(f"  разность         = {d_res['point']:+.5f}   нижняя граница = "
          f"{d_res['one_sided_lo']:+.5f}")

    print("\n--- Matched pairs: наклон внутри seed ---")
    if mp:
        print(f"  S      : {mp['point']:+.5f}  нижняя граница {mp['one_sided_lo']:+.5f}  "
              f"CI [{mp['ci'][0]:+.5f}, {mp['ci'][1]:+.5f}]  seed={mp['n_seeds']}")
    if mpn:
        print(f"  S_null : {mpn['point']:+.5f}  нижняя граница {mpn['one_sided_lo']:+.5f}")

    print("\n--- Средние по уровням ---")
    print(f"  {'w_dr':>5}{'nu':>6}{'c':>6} | {'alpha':>7}{'beta':>7} | {'S':>9}{'sem':>8} | {'S_null':>9}")
    keys = sorted({(r["w_drive"], r["nu"], r["c"]) for r in rs})
    for kk in keys:
        g = [r for r in rs if (r["w_drive"], r["nu"], r["c"]) == kk]
        S = np.array([r["S"] for r in g])
        Sn = np.array([r["S_null"] for r in g])
        Sn = Sn[np.isfinite(Sn)]
        print(f"  {kk[0]:5.1f}{kk[1]:6.2f}{kk[2]:6.1f} | "
              f"{np.mean([r['alpha'] for r in g]):7.3f}{np.mean([r['beta'] for r in g]):7.3f} | "
              f"{S.mean():+9.5f}{S.std() / np.sqrt(len(S)):8.5f} | "
              f"{(Sn.mean() if len(Sn) else float('nan')):+9.5f}")

    print("\n=== ВЕРДИКТ ПО ПРЕДРЕГИСТРИРОВАННОМУ КРИТЕРИЮ ===")
    if not (c_ok and n_ok):
        print("  ВЫВОД НЕ ВЫНОСИТСЯ: проверки валидности не пройдены.")
        return
    c1 = s_res["one_sided_lo"] > 0
    c2 = (n_res["one_sided_lo"] <= 0) and (d_res["one_sided_lo"] > 0)
    print(f"  (1) нижняя граница b_alpha > 0                 : {'ДА' if c1 else 'НЕТ'}")
    print(f"  (2) null не проходит тот же критерий, и        : "
          f"{'ДА' if c2 else 'НЕТ'}")
    print(f"      разность наклонов имеет нижнюю границу > 0")
    if c1 and c2:
        print("  ЭФФЕКТ ПОДТВЕРЖДЁН. Самовключение увеличивает остаток")
        print("  сверх замыкания контура, и это не объясняется beta.")
        if mde_val and s_res["point"] * (a.max() - a.min()) < mde_val:
            print(f"  ОГОВОРКА: эффект по размаху alpha "
                  f"({s_res['point'] * (a.max() - a.min()):.5f}) меньше MDE "
                  f"({mde_val:.5f}).")
    else:
        print("  ЭФФЕКТ НЕ ПОДТВЕРЖДЁН по предрегистрированному критерию.")
    print()


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "raw_confirm.csv")
