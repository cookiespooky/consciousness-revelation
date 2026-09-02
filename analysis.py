"""Анализ строится ИСКЛЮЧИТЕЛЬНО из сырого CSV. Систему заново не запускает.

Первичный эндпоинт зафиксирован до просмотра данных:

    Gap = b0 + b_alpha * alpha + b_beta * beta + eps,   H0: b_alpha = 0

на self-ветке при максимальной ёмкости p. beta входит ковариатой, потому что
ручка k двигает самовключение и самонаблюдаемость одновременно. Доверительный
интервал — кластерный бутстрэп по seed. Всё остальное (rho*, свип по ёмкости,
collapse-тест) объявлено вторичным и разведочным.
"""

import csv
import numpy as np

PRIMARY_THETA = 0.05
N_BOOT = 4000


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "branch" else float(v)) for k, v in r.items()})
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def paired_S(rows, eps=0.0, p=None):
    """S = Gap_self - Gap_yoked, спаренно по (seed, k, c, p).

    Первичная зависимая переменная. Замыкание контура действует на обе ветки
    одинаково и в разности сокращается; остаётся только вклад самовключения.
    """
    p = p if p is not None else max(r["p"] for r in rows)
    idx = {}
    for r in rows:
        if r["eps"] == eps and r["p"] == p and not r.get("degenerate", 0):
            key = (r["seed"], r["k"], r["c"], r.get("w_drive", 0), r.get("nu", 0))
            idx.setdefault(key, {})[r["branch"]] = r
    out = []
    for key, d in idx.items():
        if "self" in d and "yoked" in d and not d["yoked"].get("degenerate", 0):
            s = d["self"]
            rec = dict(seed=key[0], k=key[1], c=key[2], w_drive=key[3], nu=key[4],
                       p=p, alpha=s["alpha"], beta=s["beta"],
                       S=s["Gap"] - d["yoked"]["Gap"],
                       Gap=s["Gap"], gap_yoked=d["yoked"]["Gap"],
                       diverged=s["diverged"],
                       conv_max=max(s.get("conv", 0), s.get("conv_y", 0),
                                    d["self2"].get("conv", 0) if "self2" in d else 0))
            # S_null: та же разность между двумя независимыми подгонками
            # физически тождественной системы. Масштаб, который даёт процедура.
            rec["S_null"] = (s["Gap"] - d["self2"]["Gap"]) if "self2" in d else float("nan")
            out.append(rec)
    return out


def design(rs, dep="S"):
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    y = np.array([r[dep] for r in rs])
    X = np.column_stack([np.ones_like(a), a, b])
    return X, y


def cluster_bootstrap(rs, n_boot=N_BOOT, rng=None):
    """Кластерный бутстрэп по seed: пересэмплируем seed'ы, а не строки."""
    rng = rng or np.random.default_rng(4242)
    by = {}
    for r in rs:
        by.setdefault(r["seed"], []).append(r)
    seeds = list(by)
    coefs = []
    for _ in range(n_boot):
        pick = rng.choice(len(seeds), size=len(seeds), replace=True)
        sub = [r for i in pick for r in by[seeds[i]]]
        X, y = design(sub)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        coefs.append(ols(X, y))
    return np.array(coefs)


def collinearity(rs):
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def primary(rows, branch="self"):
    """Первичный тест: S = b0 + b_alpha*alpha + b_beta*beta при максимальной ёмкости."""
    pmax = max(r["p"] for r in rows)
    rs = paired_S(rows, eps=0.0, p=pmax)
    X, y = design(rs)
    beta_hat = ols(X, y)
    boot = cluster_bootstrap(rs)
    lo, hi = np.percentile(boot[:, 1], [2.5, 97.5])
    return dict(p=pmax, n=len(rs), b0=beta_hat[0], b_alpha=beta_hat[1],
                b_beta=beta_hat[2], ci=(float(lo), float(hi)),
                collinearity=collinearity(rs))


def capacity_trend(rows, branch="self"):
    """b_alpha как функция ёмкости: различает сценарии B и C."""
    out = []
    for p in sorted({r["p"] for r in rows}):
        rs = paired_S(rows, eps=0.0, p=p)
        if len(rs) < 6:
            continue
        X, y = design(rs)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        boot = cluster_bootstrap(rs, n_boot=800)
        lo, hi = np.percentile(boot[:, 1], [2.5, 97.5])
        out.append((p, float(ols(X, y)[1]), float(lo), float(hi)))
    return out


def mde_from_positive_control(rows, branch="self"):
    """Минимальный обнаружимый эффект в единицах Gap.

    Положительный контроль — калиброванная помеха eps на входе self-ветки.
    Ищем наименьший eps, чей сдвиг Gap отделим от eps=0 кластерным бутстрэпом.
    """
    pmax = max(r["p"] for r in rows)
    base = paired_S(rows, eps=0.0, p=pmax)
    if not base:
        return None
    b0 = np.mean([r["S"] for r in base])
    rng = np.random.default_rng(99)
    res = []
    for e in sorted({r["eps"] for r in rows if r["eps"] > 0}):
        rs = paired_S(rows, eps=e, p=pmax)
        if len(rs) < 4:
            continue
        d = np.array([r["S"] for r in rs]) - b0
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res.append((e, float(d.mean()), float(lo), float(hi),
                    bool(lo > 0 or hi < 0)))
    det = [r for r in res if r[4]]
    mde_gap = min(abs(r[1]) for r in det) if det else None
    # MDE получен в единицах Gap, а первичный эндпоинт — НАКЛОН по alpha.
    # Сравнивать их напрямую нельзя: нужно перевести в единицы наклона,
    # поделив на наблюдаемый размах alpha.
    al = [r["alpha"] for r in paired_S(rows, eps=0.0, p=pmax)]
    span = (max(al) - min(al)) if al else 0.0
    mde_slope = (mde_gap / span) if (mde_gap is not None and span > 1e-9) else None
    return dict(table=res, mde=mde_gap, mde_slope=mde_slope, alpha_span=span)


def collapse_test(rows, branch="self"):
    """Схлопываются ли разные механизмы на одну кривую Gap = f(alpha, beta)?

    Сравниваем R^2 модели (alpha, beta) с моделью, которой дополнительно выданы
    сами ручки k и c. Если ручки не добавляют объяснённой дисперсии, alpha и beta
    исчерпывают эффект — латентные переменные выбраны верно.
    """
    pmax = max(r["p"] for r in rows)
    rs = paired_S(rows, eps=0.0, p=pmax)
    y = np.array([r["S"] for r in rs])
    a = np.array([r["alpha"] for r in rs])
    b = np.array([r["beta"] for r in rs])
    k = np.array([r["k"] for r in rs])
    c = np.array([r["c"] for r in rs])

    def r2(X):
        w = ols(X, y)
        resid = y - X @ w
        return 1 - resid.var() / (y.var() + 1e-12)

    one = np.ones_like(y)
    return dict(r2_ab=float(r2(np.column_stack([one, a, b]))),
                r2_abkc=float(r2(np.column_stack([one, a, b, k, c]))))


def rho_gap(rows):
    """Вторичное: разность безразмерных rho* между self и external по alpha."""
    pmax = max(r["p"] for r in rows)
    out = []
    for key in sorted({(r["k"], r["c"]) for r in rows}):
        s = sel(rows, branch="self", eps=0.0, p=pmax, k=key[0], c=key[1])
        e = sel(rows, branch="external", eps=0.0, p=pmax, k=key[0], c=key[1])
        if not s or not e:
            continue
        out.append((key[0], key[1],
                    float(np.mean([r["alpha"] for r in s])),
                    float(np.mean([r["rho_star"] for r in s])
                          - np.mean([r["rho_star"] for r in e]))))
    return out


def branch_means(rows):
    pmax = max(r["p"] for r in rows)
    out = {}
    for br in ("self", "external", "self2", "yoked", "null"):
        rs = [r for r in sel(rows, branch=br, eps=0.0, p=pmax)
              if not r.get("degenerate", 0)]
        if rs:
            out[br] = (float(np.mean([r["R"] for r in rs])),
                       float(np.mean([r["Gap"] for r in rs])))
    return out


def report(path):
    rows = load(path)
    pr = primary(rows)
    mde = mde_from_positive_control(rows)
    cap = capacity_trend(rows)
    col = collapse_test(rows)
    bm = branch_means(rows)

    print(f"\n=== СЫРЫЕ ДАННЫЕ: {path}  ({len(rows)} строк) ===\n")
    print("Средние по веткам (p максимальное, eps=0):")
    print(f"  {'ветка':<10} {'R':>8} {'Gap':>9}")
    for br, (R, G) in bm.items():
        print(f"  {br:<10} {R:8.4f} {G:9.4f}")

    print("\n--- Санити-чек: нулевая система ---")
    if "null" in bm:
        ok = abs(bm["null"][1]) < 0.02
        print(f"  Gap(null) = {bm['null'][1]:+.4f}   "
              f"{'OK, установка умеет выдавать ноль' if ok else 'ТРЕВОГА: ноль не воспроизводится'}")

    print("\n--- Первичный эндпоинт: S = Gap_self - Gap_yoked ~ 1 + alpha + beta ---")
    print(f"  n = {pr['n']},  p = {pr['p']:.0f}")
    print(f"  b_alpha = {pr['b_alpha']:+.4f}   95% CI [{pr['ci'][0]:+.4f}, {pr['ci'][1]:+.4f}]")
    print(f"  b_beta  = {pr['b_beta']:+.4f}")
    print(f"  |corr(alpha,beta)| = {pr['collinearity']:.3f}"
          f"{'   ВНИМАНИЕ: коллинеарность, b_alpha неидентифицируем' if pr['collinearity'] > 0.95 else ''}")

    print("\n--- Положительный контроль -> MDE ---")
    if mde and mde["table"]:
        print(f"  {'eps':>6} {'dGap':>9} {'CI низ':>9} {'CI верх':>9}  обнаружен")
        for e, d, lo, hi, det in mde["table"]:
            print(f"  {e:6.3f} {d:+9.4f} {lo:+9.4f} {hi:+9.4f}  {'да' if det else 'нет'}")
        if mde["mde"]:
            print(f"  MDE = {mde['mde']:.4f} в единицах Gap; размах alpha = {mde['alpha_span']:.3f}")
            print(f"  => порог для наклона b_alpha: {mde['mde_slope']:.3f}")
            if all(r[1] < 0 for r in mde["table"]):
                print("  ТРЕВОГА: эффект положительного контроля ОТРИЦАТЕЛЕН при всех eps.")
                print("  Помеха на входе ослабляет рулевой сигнал G и тем самым УМЕНЬШАЕТ")
                print("  самовключение. Контроль калибрует не ту величину; MDE недействителен.")
        else:
            print("  MDE не установлен")
    else:
        print("  прогонов положительного контроля в файле нет")

    print("\n--- Свип по ёмкости: b_alpha(p) ---")
    print(f"  {'p':>5} {'b_alpha':>9} {'CI низ':>9} {'CI верх':>9}")
    for p, b, lo, hi in cap:
        print(f"  {p:5.0f} {b:+9.4f} {lo:+9.4f} {hi:+9.4f}")

    print("\n--- Collapse-тест ---")
    print(f"  R^2(alpha,beta)        = {col['r2_ab']:.4f}")
    print(f"  R^2(alpha,beta,k,c)    = {col['r2_abkc']:.4f}")
    inc = col["r2_abkc"] - col["r2_ab"]
    rel = inc / max(col["r2_ab"], 1e-9)
    print(f"  прирост от сырых ручек = {inc:+.4f}  (относительно базы: x{1 + rel:.1f})")
    print("  вердикт: " + ("СХЛОПЫВАНИЕ ЕСТЬ, alpha и beta исчерпывают эффект"
                           if rel < 0.25 else
                           "СХЛОПЫВАНИЯ НЕТ: сырые ручки несут то, чего alpha и beta не ловят"))

    print("\n--- Вторичное: delta rho* (self - external) по alpha ---")
    print(f"  {'k':>6} {'c':>6} {'alpha':>7} {'d rho*':>9}")
    for k, c, a, d in rho_gap(rows):
        print(f"  {k:6.2f} {c:6.2f} {a:7.3f} {d:+9.3f}")

    # Прямая оценка уровня S — не наклона. Здесь мощности обычно хватает.
    ps2 = paired_S(rows, eps=0.0, p=pr["p"])
    Sv = np.array([r["S"] for r in ps2])
    rng2 = np.random.default_rng(7)
    bs = np.array([Sv[rng2.integers(0, len(Sv), len(Sv))].mean() for _ in range(4000)])
    s_lo, s_hi = np.percentile(bs, [2.5, 97.5])
    print("\n--- Прямая оценка уровня S (не наклона) ---")
    print(f"  mean S = {Sv.mean():+.5f}   95% CI [{s_lo:+.5f}, {s_hi:+.5f}]   n = {len(Sv)}")
    if mde and mde["mde"]:
        inside = max(abs(s_lo), abs(s_hi)) <= mde["mde"]
        print(f"  MDE = {mde['mde']:.4f}  ->  " + ("CI внутри порога: эффект самовключения "
              "на уровне S не превышает порога обнаружения"
              if inside else "CI выходит за порог: вывод об уровне S не выносится"))

    print("\n=== КЛАССИФИКАЦИЯ СЦЕНАРИЯ ===")
    lo, hi = pr["ci"]
    sig = lo > 0 or hi < 0
    m = mde.get("mde_slope") if mde else None
    bad_pc = bool(mde and mde["table"] and all(r[1] < 0 for r in mde["table"]))
    if pr["collinearity"] > 0.85 or bad_pc:
        print("  ВЫВОД НЕ ВЫНОСИТСЯ. Предпосылки первичного теста нарушены:")
        if pr["collinearity"] > 0.85:
            print(f"    - коллинеарность alpha/beta = {pr['collinearity']:.3f}: b_alpha "
                  "неидентифицируем, знак и величина недостоверны")
        if bad_pc:
            print("    - положительный контроль имеет обратный знак: MDE недействителен")
        print("  Ни A, ни B, ни C. Сначала чинить протокол, потом классифицировать.")
    elif not sig and m is not None and max(abs(lo), abs(hi)) <= m:
        # Эквивалентность объявляется, только если в порог укладывается ВЕСЬ
        # доверительный интервал. Малой точечной оценки недостаточно: широкий CI
        # означает нехватку мощности, а не отсутствие эффекта.
        print("  A — NO. Структурный эффект самовключения не обнаружен.")
        print(f"      Весь CI наклона укладывается в порог +-{m:.4f}.")
        print("      Структурная ветка закрыта; переходить к precision / D_KL.")
    elif not sig:
        print("  НЕОПРЕДЕЛЁННО — тест недостаточно мощен для вывода.")
        print(f"      CI наклона [{lo:+.3f}, {hi:+.3f}] шире порога "
              f"+-{m:.4f} в {max(abs(lo), abs(hi)) / m:.0f} раз." if m else "")
        print("      Ноль не отвергнут, но и эквивалентность не установлена.")
        print("      Нужен больший рычаг по alpha либо больше seed.")
    elif len(cap) >= 2 and cap[-1][1] < 0.4 * cap[0][1]:
        print("  B — эффект есть, но убывает с ростом ёмкости.")
        print("      Это аппроксимационная недостаточность, не структура.")
    else:
        print("  C — эффект есть и НЕ устраняется ростом ёмкости.")
        print("      Кандидат на структурный остаток самовключения.")
    print()


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "raw.csv")
