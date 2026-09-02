"""Граница существования устойчивого режима: k_c и lambda_max для self и yoked.

Вопрос один: делает ли самовключение замкнутый контур менее устойчивым, чем
точно такой же контур без самореференции.

    k_c      = max{k : попеременная подгонка сходится}
    dk_c     = k_c(self) - k_c(yoked)
    lambda   = старший показатель Ляпунова замкнутой динамики с замороженной G

Критерий сходимости зафиксирован ДО прогона и не зависит от невязки: медиана
последних трёх относительных изменений весов ниже TAU. Это свойство процедуры
подгонки, а не результата, поэтому подкрутить его под ответ нельзя.

lambda_max нужен, чтобы увидеть, КАК теряется устойчивость. Самовключённая и
yoked-системы могут иметь разную локальную динамику при одинаковых размерности
и ёмкости; если lambda_self ~ lambda_yoked вплоть до порога, структурная
гипотеза получает удар независимо от того, что показал k_c.
"""

import numpy as np

import dynamics as dyn
from dynamics import N_B, N_G, N_W, Params, State, step
from model import Basis, fit_closed_loop

TAU = 0.25          # порог сходимости по относительному изменению весов
N_TAIL = 3          # сколько последних итераций усредняем
H_FD = 1e-6         # шаг конечной разности для якобиан-вектора
T_LYAP = 400        # длина накопления показателя Ляпунова


def converged(hist):
    """Предзаданный критерий: медиана последних N_TAIL изменений весов < TAU."""
    if len(hist) < N_TAIL:
        return False
    return bool(np.median(hist[-N_TAIL:]) < TAU)


def _pack(s):
    return np.concatenate([s.w, s.w_prev, s.b, s.b_prev, s.x_g])


_I_B = 2 * N_W
_I_BP = _I_B + N_B
_I_G = _I_BP + N_B


def _unpack(v, proto):
    s = proto.copy()
    s.w = v[:N_W].copy()
    s.w_prev = v[N_W:2 * N_W].copy()
    s.b = v[_I_B:_I_BP].copy()
    s.b_prev = v[_I_BP:_I_G].copy()
    s.x_g = v[_I_G:_I_G + N_G].copy()
    return s


def lyapunov_max(par, G, Gp, branch, seed_offset=31337):
    """Старший показатель Ляпунова замкнутой динамики с замороженной G.

    Шум занулён: показатель считается на детерминированном скелете, как принято.
    Якобиан не выписывается явно — используется конечная разность вдоль вектора
    касательной (метод Беннетина с перенормировкой на каждом шаге).
    """
    p0 = par.clone(noise=0.0, prop_noise=0.0)
    rng = np.random.default_rng(seed_offset + par.seed)
    st = State(p0, rng)
    for _ in range(150):
        _, _, st = step(st, p0, G, Gp, rng, branch)

    def phi(vec):
        s = _unpack(vec, st)
        _, _, nxt = step(s, p0, G, Gp, rng, branch)
        return _pack(nxt)

    # Касательный вектор ограничен подпространством, КУДА МОДЕЛЬ ПИШЕТ: телесная
    # координата, её задержка и регистры самомодели. Мир (w1, w2) автономен, его
    # блок якобиана не зависит от остальных, поэтому полный старший показатель
    # принадлежал бы предельному циклу мира и на k не реагировал бы вовсе.
    # Треугольность блока гарантирует, что возмущение из этого подпространства
    # его не покидает.
    x = _pack(st)
    mask = np.zeros_like(x)
    mask[_I_B:_I_G + N_G] = 1.0           # тело, его задержка и регистры
    v = rng.normal(0, 1, size=x.shape) * mask
    v /= np.linalg.norm(v)
    acc, n = 0.0, 0
    for _ in range(T_LYAP):
        fx = phi(x)
        jv = (phi(x + H_FD * v) - fx) / H_FD
        nrm = float(np.linalg.norm(jv))
        if not np.isfinite(nrm) or nrm <= 0:
            return float("nan")
        acc += np.log(nrm)
        n += 1
        v = (jv * mask) / nrm
        x = fx
        if not np.all(np.isfinite(x)):
            return float("nan")
    return float(acc / n)


# Ветка "self2" физически тождественна "self" и отличается ТОЛЬКО seed'ом
# случайного базиса. Она даёт null-уровень для dk_c: разность двух независимых
# подгонок одной и той же системы. Без неё нельзя отличить эффект самовключения
# от обычного разброса между подгонками.
BRANCH_PHYS = {"self": "self", "self2": "self", "yoked": "yoked"}
BASIS_OFFSET = {"self": 0, "self2": 12345, "yoked": 0}


def _fit_at(branch, seed, c, k, p_cap, basis, T_train, n_iter, lam,
            w_drive=1.5, nu=0.55):
    par = Params(k=float(k), c=c, seed=seed, w_drive=w_drive, nu=nu)
    phys = BRANCH_PHYS[branch]
    G, Gp, hist = fit_closed_loop(par, basis, phys, T_train, n_iter, lam,
                                  twin=(phys == "yoked"))
    return par, G, Gp, hist


def refine_kc(branch, seed, c, p_cap, basis, k_lo, k_hi, T_train, n_iter,
              lam=1e-6, n_bisect=4, w_drive=1.5, nu=0.55):
    """Уточнение порога бисекцией между последним сошедшимся и первым разошедшимся.

    Шаг сетки 0.1 ограничивал бы разрешение dk_c именно на масштабе самого
    ожидаемого эффекта. Четыре деления доводят разрешение до ~0.006.
    """
    for _ in range(n_bisect):
        mid = 0.5 * (k_lo + k_hi)
        _, _, _, hist = _fit_at(branch, seed, c, mid, p_cap, basis, T_train,
                                n_iter, lam, w_drive, nu)
        if converged(hist):
            k_lo = mid
        else:
            k_hi = mid
    return k_lo


def scan_branch(branch, seed, c, p_cap, k_grid, T_train=500, n_iter=6, lam=1e-6,
                w_drive=1.5, nu=0.55):
    """Скан по k для одной ветки. Возвращает строки и уточнённое k_c."""
    basis = Basis(p_cap, seed + BASIS_OFFSET[branch])
    rows, ok_run = [], True
    k_c, k_first_bad = float("nan"), float("nan")
    for k in k_grid:
        par, G, Gp, hist = _fit_at(branch, seed, c, k, p_cap, basis, T_train,
                                   n_iter, lam, w_drive, nu)
        cv = converged(hist)
        lm = lyapunov_max(par, G, Gp, BRANCH_PHYS[branch])
        rows.append(dict(seed=seed, c=c, p=p_cap, branch=branch, k=float(k),
                         w_drive=w_drive, nu=nu, converged=int(cv), lam_max=lm,
                         conv_stat=float(np.median(hist[-N_TAIL:]))))
        # k_c — наибольшее k, до которого сходимость держалась НЕПРЕРЫВНО.
        # Так шумное «возвращение» сходимости выше порога не завышает оценку.
        if ok_run and cv:
            k_c = float(k)
        elif ok_run and not cv:
            k_first_bad = float(k)
            ok_run = False
    if np.isfinite(k_c) and np.isfinite(k_first_bad):
        k_c = refine_kc(branch, seed, c, p_cap, basis, k_c, k_first_bad,
                        T_train, n_iter, lam, w_drive=w_drive, nu=nu)
    return rows, k_c


def run_pair(args):
    """Пара веток при общих seed, c, p. Третий элемент args задаёт контраст."""
    seed, c, p_cap, k_grid, T_train, n_iter = args[:6]
    br_b = args[6] if len(args) > 6 else "yoked"
    w_drive, nu = (args[7], args[8]) if len(args) > 8 else (1.5, 0.55)
    rs, kc_s = scan_branch("self", seed, c, p_cap, k_grid, T_train, n_iter,
                           w_drive=w_drive, nu=nu)
    ry, kc_y = scan_branch(br_b, seed, c, p_cap, k_grid, T_train, n_iter,
                           w_drive=w_drive, nu=nu)

    # alpha измеряется на пороге self — там, где вопрос об устойчивости и стоит.
    from experiment import measure_alpha_beta
    a = b = float("nan")
    if np.isfinite(kc_s):
        par = Params(k=kc_s, c=c, seed=seed, w_drive=w_drive, nu=nu)
        basis = Basis(p_cap, seed)
        G, _, _ = fit_closed_loop(par, basis, "self", T_train, n_iter)
        a, b = measure_alpha_beta(par, G, None, "self")

    for r in rs + ry:
        r["k_c_self"], r["k_c_yoked"] = kc_s, kc_y
        r["alpha_at_kc"], r["beta_at_kc"] = a, b
    return rs + ry


FIELDS = ["seed", "c", "p", "w_drive", "nu", "branch", "k", "converged",
          "lam_max", "conv_stat", "k_c_self", "k_c_yoked",
          "alpha_at_kc", "beta_at_kc"]
