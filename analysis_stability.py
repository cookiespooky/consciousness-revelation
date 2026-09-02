"""Анализ границы устойчивости. Строится только из raw_stability.csv.

Первичный эндпоинт: dk_c = k_c(self) - k_c(yoked), спаренно по seed.
Отрицательное dk_c означает, что самовключение сдвигает потерю устойчивости
РАНЬШЕ, то есть подтверждает структурную гипотезу.

Вторичное: lambda_max(k) для обеих веток — как именно теряется устойчивость.
"""

import csv
import sys

import numpy as np

BISECT_RES = 0.1 / 2 ** 4      # разрешение бисекции по k


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "branch" else float(v))
                         for k, v in r.items()})
    return rows


def pairs(rows):
    """По одной записи на (seed, c): k_c обеих веток и alpha на пороге."""
    seen = {}
    for r in rows:
        key = (r["seed"], r["c"])
        if key not in seen:
            seen[key] = dict(seed=r["seed"], c=r["c"],
                             kc_self=r["k_c_self"], kc_yoked=r["k_c_yoked"],
                             alpha=r["alpha_at_kc"], beta=r["beta_at_kc"])
    out = [v for v in seen.values()
           if np.isfinite(v["kc_self"]) and np.isfinite(v["kc_yoked"])]
    for v in out:
        v["dkc"] = v["kc_self"] - v["kc_yoked"]
    return out


def boot_mean(x, n=8000, seed=11):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), tuple(np.percentile(bs, [2.5, 97.5]))


def lam_curves(rows):
    out = {}
    for br in ("self", "yoked"):
        by = {}
        for r in rows:
            if r["branch"] == br:
                by.setdefault(r["k"], []).append(r["lam_max"])
        out[br] = sorted((k, float(np.nanmean(v)), float(np.nanstd(v) / max(np.sqrt(len(v)), 1)))
                         for k, v in by.items())
    return out


def conv_curves(rows):
    out = {}
    for br in ("self", "yoked"):
        by = {}
        for r in rows:
            if r["branch"] == br:
                by.setdefault(r["k"], []).append(r["converged"])
        out[br] = sorted((k, float(np.mean(v))) for k, v in by.items())
    return out


def compare_to_null(path_effect, path_null):
    """Сравнение контраста self-vs-yoked с null-уровнем self-vs-self2.

    Null задаёт масштаб, который даёт САМА процедура: две независимые подгонки
    физически тождественной системы, отличающиеся только seed'ом базиса.
    Эффект имеет смысл только если он выходит за этот масштаб.
    """
    de = np.array([p["dkc"] for p in pairs(load(path_effect))])
    dn = np.array([p["dkc"] for p in pairs(load(path_null))])
    rng = np.random.default_rng(3)
    bs = np.array([de[rng.integers(0, len(de), len(de))].mean()
                   - dn[rng.integers(0, len(dn), len(dn))].mean()
                   for _ in range(8000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    floor = float(np.percentile(np.abs(dn), 95))
    print("\n=== ЭФФЕКТ ПРОТИВ NULL-УРОВНЯ ===")
    print(f"  self vs yoked : dk_c = {de.mean():+.4f}  (n={len(de)})")
    print(f"  self vs self2 : dk_c = {dn.mean():+.4f}  (n={len(dn)})  <- null")
    print(f"  разность      = {de.mean() - dn.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  порог обнаружения (95-й перцентиль |null|) = {floor:.4f}")
    if lo > 0 or hi < 0:
        print("  Контраст ОТЛИЧАЕТСЯ от null: эффект самовключения есть.")
    else:
        print("  Контраст НЕ отличается от null.")
        print(f"  |эффект| = {abs(de.mean()):.4f} против порога {floor:.4f}:")
        print("  измеренный сдвиг целиком лежит внутри разброса, который даёт")
        print("  простая смена seed'а базиса при физически той же системе.")
    return float(de.mean()), float(dn.mean()), floor


def report(path):
    rows = load(path)
    ps = pairs(rows)
    print(f"\n=== ГРАНИЦА УСТОЙЧИВОСТИ: {path} ({len(rows)} строк, {len(ps)} пар) ===\n")

    ks, ky = [p["kc_self"] for p in ps], [p["kc_yoked"] for p in ps]
    d = [p["dkc"] for p in ps]
    ms, cis = boot_mean(ks)
    my, ciy = boot_mean(ky)
    md, cid = boot_mean(d)

    print("--- Критические k ---")
    print(f"  k_c(self)  = {ms:.4f}  95% CI [{cis[0]:.4f}, {cis[1]:.4f}]")
    print(f"  k_c(yoked) = {my:.4f}  95% CI [{ciy[0]:.4f}, {ciy[1]:.4f}]")

    print("\n--- ПЕРВИЧНЫЙ ЭНДПОИНТ: dk_c = k_c(self) - k_c(yoked) ---")
    print(f"  dk_c = {md:+.4f}  95% CI [{cid[0]:+.4f}, {cid[1]:+.4f}]  n = {len(d)}")
    sig = cid[0] > 0 or cid[1] < 0
    if sig and md < 0:
        print("  ЗНАЧИМО И ОТРИЦАТЕЛЬНО: самовключение сдвигает потерю")
        print("  устойчивости раньше — структурная гипотеза подтверждается.")
    elif sig and md > 0:
        print("  ЗНАЧИМО, НО ПОЛОЖИТЕЛЬНО: самовключённая система устойчивее")
        print("  yoked. Это противоположно гипотезе и требует объяснения.")
    else:
        w = max(abs(cid[0]), abs(cid[1]))
        print(f"  Ноль не отвергнут. Полуширина CI = {w:.4f}, "
              f"разрешение бисекции = {BISECT_RES:.4f}.")
        print("  ВНИМАНИЕ: эквивалентность объявлять нельзя без null-контроля")
        print("  self-vs-self (разные basis seed). Сейчас его в данных нет,")
        print("  поэтому нулевой уровень dk_c неизвестен.")

    if len(ps) > 3:
        a = np.array([p["alpha"] for p in ps])
        y = np.array([p["dkc"] for p in ps])
        ok = np.isfinite(a) & np.isfinite(y)
        if ok.sum() > 3 and a[ok].std() > 1e-9:
            X = np.column_stack([np.ones(ok.sum()), a[ok]])
            b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            rng = np.random.default_rng(5)
            bs = []
            for _ in range(4000):
                i = rng.integers(0, ok.sum(), ok.sum())
                bs.append(np.linalg.lstsq(X[i], y[ok][i], rcond=None)[0][1])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"\n--- dk_c = f(alpha) ---")
            print(f"  наклон = {b[1]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"   размах alpha = {a[ok].max() - a[ok].min():.3f}")

    print("\n--- Доля сошедшихся по k ---")
    cc = conv_curves(rows)
    print(f"  {'k':>5} {'self':>8} {'yoked':>8}")
    dy = dict(cc["yoked"])
    for k, v in cc["self"]:
        print(f"  {k:5.2f} {v:8.2f} {dy.get(k, float('nan')):8.2f}")

    print("\n--- lambda_max(k), ведомое подпространство ---")
    lc = lam_curves(rows)
    ly = {k: (m, s) for k, m, s in lc["yoked"]}
    print(f"  {'k':>5} {'self':>9} {'sem':>7} {'yoked':>9} {'sem':>7}")
    for k, m, s in lc["self"]:
        m2, s2 = ly.get(k, (float("nan"), float("nan")))
        print(f"  {k:5.2f} {m:+9.4f} {s:7.4f} {m2:+9.4f} {s2:7.4f}")
    print()


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "raw_stability.csv")
    if len(sys.argv) > 2:
        compare_to_null(sys.argv[1], sys.argv[2])
