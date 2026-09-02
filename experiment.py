"""Один прогон конфигурации: четыре ветки, интервенционные alpha/beta, R, rho*.

Ветки
-----
self      G предсказывает (x_op, x_g) — множество, включающее её собственные
          регистры. Контур замыкает сама G.
yoked     G предсказывает (x_op, x_gp) — регистры двойника G'. Та же
          размерность, та же форма динамики, самовключения второго блока нет.
external  O предсказывает ровно то же, что и self, на той же траектории, с той
          же ёмкостью и тем же входом, но контур замыкает G, а не O.
          O не является частью моделируемого объекта.
null      k = 0 и цель (x_op, x_dec): причинного контакта G с целью нет вовсе.
          Установка обязана выдать здесь Gap ~ 0.

Первичная зависимая переменная: S = Gap_self - Gap_yoked.

Не Gap_self: smoke-прогон показал Gap_yoked ~ 0.9 * Gap_self, то есть почти весь
рост невязки даёт САМО ЗАМЫКАНИЕ КОНТУРА — модель влияет на динамику, которую
затем наблюдает, — а не включение её собственной реализации в моделируемое
множество. Разность S вычитает эффект замыкания и оставляет только вклад
самовключения. S ~ 0 на всём диапазоне при разведённых alpha/beta означает, что
структурная гипотеза самовключения не работает.
"""

import numpy as np

import dynamics as dyn
from dynamics import N_B, N_G, N_OP, N_W, N_Y, Params, State, build_input, step
from model import Basis, Predictor, fit_closed_loop, nrmse, ridge

# --- Набор интервенций: зафиксирован ДО запуска, одинаков для всех конфигураций.
_INT_RNG = np.random.default_rng(2718281)
N_INT_DIR = 5            # число направлений вмешательства
N_INT_TIME = 50          # число моментов приложения
INT_SCALE = 0.10         # АБСОЛЮТНАЯ величина возмущения, одна для обоих плеч.
# Прежде она масштабировалась на sigma каждого блока в отдельности, из-за чего
# слабо меняющийся yhat возмущался слабо и числитель alpha занижался: при
# w_drive=0, то есть в режиме чистого самоконтура, alpha давала 0.084 вместо
# ожидаемой почти единицы. Кроме того, распределение интервенций тогда зависело
# от конфигурации, а оно должно быть одним для всех.
_DIR_Y = _INT_RNG.normal(0, 1, size=(N_INT_DIR, N_Y))
_DIR_Y /= np.linalg.norm(_DIR_Y, axis=1, keepdims=True)
_DIR_OP = _INT_RNG.normal(0, 1, size=(N_INT_DIR, N_OP))
_DIR_OP /= np.linalg.norm(_DIR_OP, axis=1, keepdims=True)

RHO_GRID = np.logspace(-2.2, 1.2, 44)
THETA_CONSIST = 0.95


def _yhat_of(st, p, G, rng_state):
    """Невозмущённый выход модели в данном состоянии."""
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    z = build_input(st.w, st.w_prev, st.b, st.b_prev, st.x_g, p, rng)
    return G.predict(z)


def _one_step_pair(st, p, G, Gp, branch, rng_state, block, delta):
    """Фактический и контрфактический шаг из одного состояния.

    Общие случайные числа: rng восстанавливается перед контрфактическим шагом,
    поэтому eta и шум проприоцепции совпадают. Разница вызвана только do().
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    z_f, _, nx_f = step(st, p, G, Gp, rng, branch)

    st2, dy, force = st.copy(), None, None
    if block == "yhat":
        dy = delta                      # do(yhat): внутренняя переменная G
    else:
        # do(x_op): мир и тело вместе, ПУТЬ ЧЕРЕЗ G ЗАБЛОКИРОВАН — выход модели
        # держится на невозмущённом значении.
        st2.w = st2.w + delta[:N_W]
        st2.b = st2.b + delta[N_W:]
        force = _yhat_of(st, p, G, rng_state)
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = rng_state
    _, _, nx_c = step(st2, p, G, Gp, rng2, branch, dyhat=dy, force_yhat=force)

    # Вход самомодели на следующем такте — то, что видит G о самой себе.
    rng3 = np.random.default_rng()
    rng3.bit_generator.state = rng_state
    z_next_f = build_input(nx_f.w, nx_f.w_prev, nx_f.b, nx_f.b_prev, nx_f.x_g, p, rng3)
    rng4 = np.random.default_rng()
    rng4.bit_generator.state = rng_state
    z_next_c = build_input(nx_c.w, nx_c.w_prev, nx_c.b, nx_c.b_prev, nx_c.x_g, p, rng4)

    dY = np.linalg.norm(nx_c.target(branch) - nx_f.target(branch))
    dZ = np.linalg.norm(z_next_c - z_next_f)
    return dY, dZ, nx_f


def measure_alpha_beta(p, G, Gp, branch, T=250, seed_offset=555):
    """Интервенционная оценка самовключения (alpha) и самонаблюдаемости (beta).

        alpha = влияние выходного регистра G на ЦЕЛЬ
                -------------------------------------------------------
                то же + влияние «всего остального» ПРИ ЗАБЛОКИРОВАННОМ G

    Плечо знаменателя считается путь-специфично: при возмущении мира и тела
    выход модели фиксируется на невозмущённом значении. Иначе возмущение мира
    доходило бы до цели ЧЕРЕЗ G, знаменатель раздувался бы тем же самовключением,
    которое измеряется, и alpha парадоксально падала бы при усилении самоконтура
    (при w_drive=0 она давала 0.02 вместо ожидаемой почти единицы).

        beta  = влияние выходного регистра G на ВХОД G / полное влияние на вход

    Вмешательство идёт в yhat — переменную, причинно входящую в предсказываемую
    величину. Возмущать скрытые регистры x_g бессмысленно: при c=0 пути от них к
    цели вообще нет, и alpha перестал бы зависеть от k, хотя эффект есть.

    Обе — доли в [0,1]. Ни та ни другая не задаются рукой: ручки — только k и c.
    """
    rng = np.random.default_rng(770_000 + 13 * p.seed + seed_offset)
    st = State(p, rng)
    for _ in range(60):
        _, _, st = step(st, p, G, Gp, rng, branch)

    times = np.linspace(0, T - 1, N_INT_TIME).astype(int)
    acc = {"yhat": [0.0, 0.0], "op": [0.0, 0.0]}
    n = 0
    tset = set(times.tolist())
    for t in range(T):
        rs = rng.bit_generator.state
        if t in tset:
            for j in range(N_INT_DIR):
                dY, dZ, _ = _one_step_pair(st, p, G, Gp, branch, rs,
                                           "yhat", INT_SCALE * _DIR_Y[j])
                acc["yhat"][0] += dY
                acc["yhat"][1] += dZ
                dY, dZ, _ = _one_step_pair(st, p, G, Gp, branch, rs,
                                           "op", INT_SCALE * _DIR_OP[j])
                acc["op"][0] += dY
                acc["op"][1] += dZ
                n += 1
        _, _, st = step(st, p, G, Gp, rng, branch)

    eG_y, eG_z = acc["yhat"][0] / n, acc["yhat"][1] / n
    eO_y, eO_z = acc["op"][0] / n, acc["op"][1] / n
    alpha = eG_y / (eG_y + eO_y + 1e-12)
    beta = eG_z / (eG_z + eO_z + 1e-12)
    return float(alpha), float(beta)


def rho_star(Yhat, Y, sigma_dx, theta=THETA_CONSIST):
    """Минимальное безразмерное разрешение, при котором самосогласованность > theta.

    Самосогласовано <=> предсказание и реализовавшееся состояние попали в одну
    ячейку решётки по ВСЕМ компонентам. Разрешение нормировано на масштаб
    одношагового смещения той же системы, иначе rho* мерил бы диаметр аттрактора.
    """
    for rt in RHO_GRID:
        cell = rt * sigma_dx
        ok = np.all(np.floor(Yhat / cell) == np.floor(Y / cell), axis=1)
        if ok.mean() > theta:
            return float(rt)
    return float(RHO_GRID[-1] * 2)


def _sigma_dx(Y):
    return float(np.mean(np.std(np.diff(Y, axis=0), axis=0)) + 1e-12)


# Порог вырождения. Если разброс цели опускается до уровня шума динамики,
# предсказывать в ней нечего и нормированная невязка теряет смысл: при
# w_drive=0 у тела не остаётся ни одного экзогенного источника вариативности,
# все ветки дают R ~ 1.0, а null-ветка выдаёт бессмысленные значения.
# Такие конфигурации помечаются и исключаются в анализе.
DEGEN_FLOOR = 3.0        # в единицах шума динамики


def _degenerate(Y, p):
    return int(float(np.mean(np.std(Y, axis=0))) < DEGEN_FLOOR * p.noise)


def _dg(Y, p):
    """Флаг и разброс цели ДЛЯ КОНКРЕТНОЙ ВЕТКИ.

    Считать флаг по цели self недостаточно: у null-ветки k=0, и при слабом
    экзогенном приводе её цель схлопывается, когда self ещё вполне жива.
    """
    return dict(degenerate=_degenerate(Y, p),
                std_y=float(np.mean(np.std(Y, axis=0))))


def run_config(k, c, p_cap, seed, eps=0.0, T_train=1200, T_test=1200,
               n_iter=5, lam=1e-6, w_drive=1.5, nu=0.55):
    """Полная конфигурация -> список строк сырых данных (по одной на ветку)."""
    par = Params(k=k, c=c, eps=eps, seed=seed, w_drive=w_drive, nu=nu)
    basis = Basis(p_cap, seed)
    rows = []

    # ---- self -------------------------------------------------------------
    G, _, hist = fit_closed_loop(par, basis, "self", T_train, n_iter, lam)
    Z, Yh, Y = dyn.rollout(par, G, None, "self", T_test, seed_offset=9001)
    R_self = nrmse(Yh, Y)
    sdx = _sigma_dx(Y)
    rho_self = rho_star(Yh, Y, sdx)
    a_self, b_self = measure_alpha_beta(par, G, None, "self")
    conv = hist[-1] if hist else float("nan")

    # ---- external ---------------------------------------------------------
    # Контур замыкает та же замороженная G. O учится на отдельной траектории
    # той же системы и предсказывает то же самое, не участвуя в динамике.
    Zo, _, Yo = dyn.rollout(par, G, None, "self", T_train, seed_offset=9500)
    basis_o = Basis(p_cap, seed + 77)
    O = Predictor(basis_o, ridge(basis_o(Zo), Yo, lam))
    R_ext = nrmse(O.predict_batch(Z), Y)
    rho_ext = rho_star(O.predict_batch(Z), Y, sdx)

    # ---- self2 ------------------------------------------------------------
    # Физически тождественна self, отличается ТОЛЬКО seed'ом случайного базиса.
    # Даёт null-уровень для S: разброс между двумя независимыми подгонками одной
    # и той же системы. В прогоне устойчивости этот контроль оказался решающим —
    # он давал сдвиг больше самого измеряемого эффекта.
    basis2 = Basis(p_cap, seed + 12345)
    G2, _, hist2 = fit_closed_loop(par, basis2, "self", T_train, n_iter, lam)
    Z2, Yh2, Y2 = dyn.rollout(par, G2, None, "self", T_test, seed_offset=9004)
    R_self2 = nrmse(Yh2, Y2)
    Zo2, _, Yo2 = dyn.rollout(par, G2, None, "self", T_train, seed_offset=9700)
    basis_o2 = Basis(p_cap, seed + 77)
    O2 = Predictor(basis_o2, ridge(basis_o2(Zo2), Yo2, lam))
    R_ext2 = nrmse(O2.predict_batch(Z2), Y2)
    rho_self2 = rho_star(Yh2, Y2, _sigma_dx(Y2))

    # ---- yoked ------------------------------------------------------------
    # Положительный контроль калибрует S, поэтому помеха применяется ТОЛЬКО к
    # self-ветке. Прошлая версия пускала eps через общий rollout, и в разности
    # S = Gap_self - Gap_yoked она сокращалась начисто — контроль ничего не мерил.
    par_y = par.clone(eps=0.0)
    Gy, Gpy, hist_y = fit_closed_loop(par_y, basis, "yoked", T_train, n_iter,
                                      lam, twin=True)
    conv_y = hist_y[-1] if hist_y else float("nan")
    Zy, Yhy, Yy = dyn.rollout(par_y, Gy, Gpy, "yoked", T_test, seed_offset=9002)
    R_yok = nrmse(Yhy, Yy)
    rho_yok = rho_star(Yhy, Yy, _sigma_dx(Yy))
    a_yok, b_yok = measure_alpha_beta(par_y, Gy, Gpy, "yoked")

    # ---- null -------------------------------------------------------------
    # k = 0 и цель без причинного контакта с G. Проверка, что установка умеет
    # выдать ноль.
    par0 = par.clone(k=0.0)
    Gn, _, _ = fit_closed_loop(par0, basis, "null", T_train, n_iter, lam)
    Zn, Yhn, Yn = dyn.rollout(par0, Gn, None, "null", T_test, seed_offset=9003)
    R_null = nrmse(Yhn, Yn)
    Zno, _, Yno = dyn.rollout(par0, Gn, None, "null", T_train, seed_offset=9600)
    basis_n = Basis(p_cap, seed + 77)
    On = Predictor(basis_n, ridge(basis_n(Zno), Yno, lam))
    R_null_ext = nrmse(On.predict_batch(Zn), Yn)
    # У нулевой ветки причинного контакта G с целью нет по построению,
    # поэтому alpha = 0 аналитически и интервенции на неё не расходуются.
    a_null, b_null = 0.0, 0.0

    # Флаг расходимости попеременной подгонки. Критерий зафиксирован ДО запуска и
    # не зависит от Gap: это свойство процедуры подгонки, а не результата.
    diverged = int(conv > 1.0)
    common = dict(seed=seed, k=k, c=c, p=p_cap, eps=eps, w_drive=w_drive, nu=nu,
                  sigma_dx=sdx, conv=conv, diverged=diverged, conv_y=conv_y)
    rows.append(dict(common, branch="self", alpha=a_self, beta=b_self,
                     R=R_self, rho_star=rho_self, Gap=R_self - R_ext, **_dg(Y, par)))
    rows.append(dict(common, branch="external", alpha=a_self, beta=b_self,
                     R=R_ext, rho_star=rho_ext, Gap=0.0, **_dg(Y, par)))
    rows.append(dict(common, branch="self2", alpha=a_self, beta=b_self,
                     R=R_self2, rho_star=rho_self2, Gap=R_self2 - R_ext2,
                     **_dg(Y2, par)))
    rows.append(dict(common, branch="yoked", alpha=a_yok, beta=b_yok,
                     R=R_yok, rho_star=rho_yok, Gap=R_yok - R_ext, **_dg(Yy, par)))
    rows.append(dict(common, branch="null", alpha=a_null, beta=b_null,
                     R=R_null, rho_star=rho_star(Yhn, Yn, _sigma_dx(Yn)),
                     Gap=R_null - R_null_ext, **_dg(Yn, par0)))
    return rows


FIELDS = ["seed", "k", "c", "p", "eps", "w_drive", "nu", "branch", "alpha",
          "beta", "R", "Gap", "rho_star", "sigma_dx", "conv", "diverged",
          "conv_y", "degenerate", "std_y"]
