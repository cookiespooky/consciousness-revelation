"""Физика установки v2: мир / тело / регистры, с управляемой долей самовключения.

Отличие от v1. Там мир входил в цель и был автономен, поэтому в знаменателе
alpha всегда доминировал он, и alpha упиралась в потолок ~0.25. Здесь мир
остаётся незамораживаемым источником вариативности, но ИЗ ЦЕЛИ УБРАН. Цель —
тело плюс регистры самомодели, то есть ровно та часть состояния, причинность
которой делят между собой внешний источник и сама модель.

Блоки состояния:

    w      (N_W)  автономный предельный цикл. Выход G на него не влияет вообще:
                  систему нельзя заморозить, задачу нельзя тривиализовать.
                  В цель НЕ входит.
    b      (N_B)  тело. Причинность делят три источника:
                    nu * b       — собственная инерция
                    w_drive * .. — экзогенный привод от мира
                    k * H(yhat)  — руль самомодели
    x_g    (N_G)  регистры, В КОТОРЫХ РЕАЛИЗОВАНА самомодель G
    x_gp   (N_G)  регистры двойника G' (yoked): та же размерность и форма, но
                  его выход в состояние не возвращается
    x_dec  (N_G)  декоративный блок нулевой ветки: причинного контакта с G нет

Цель:  y = (b, x_g)  |  yoked: (b, x_gp)  |  null: (b, x_dec)

Отсюда управление самовключением: alpha ~ k / (k + w_drive + nu). Уменьшая
экзогенную причинность тела, можно довести alpha почти до единицы, НЕ загоняя k
в область, где подгонка заведомо расходится. alpha при этом по-прежнему
измеряется интервенционно, а не задаётся.

Порядок такта строго фиксирован:

    z_t       = (w, w_prev, b, b_prev, c * зашумлённый x_g)   телеметрии нет
    yhat_t    = G(z_t)
    w[t+1]    = автономный поворот с притяжением к r=1
    b[t+1]    = nu*b + w_drive*tanh(Wb w) + k*tanh(B yhat) + eta
    x_g[t+1]  = (1-gamma) x_g + gamma * tanh(A yhat)
"""

import numpy as np

N_W = 2                    # автономный мир, в цель не входит
N_B = 3                    # тело
N_G = 2                    # регистры самомодели
N_Y = N_B + N_G            # размерность цели
N_OP = N_W + N_B           # «всё остальное» для знаменателя alpha
DIM_Z = 2 * N_W + 2 * N_B + N_G


class Params:
    """Все константы установки. Фиксируются до запуска и не трогаются."""

    def __init__(self, k=0.0, c=0.0, eps=0.0, seed=0,
                 theta=0.35, mu=0.30, gamma=0.40, nu=0.55, w_drive=1.5,
                 noise=0.02, prop_noise=0.05):
        self.k = float(k)                 # ручка: сила руля самомодели
        self.c = float(c)                 # ручка: усиление проприоцептивного канала
        self.w_drive = float(w_drive)     # ручка: экзогенный привод тела от мира
        self.nu = float(nu)               # ручка: инерция тела
        self.eps = float(eps)             # положительный контроль, см. rollout()
        self.seed = int(seed)
        self.theta = theta
        self.mu = mu
        self.gamma = gamma
        self.noise = noise
        self.prop_noise = prop_noise

        rng = np.random.default_rng(10_000 + self.seed)
        self.B = rng.normal(0, 1.0, size=(N_B, N_Y)) / np.sqrt(N_Y)   # руль в тело
        self.A = rng.normal(0, 1.0, size=(N_G, N_Y)) / np.sqrt(N_Y)   # запись в регистры
        self.Wb = rng.normal(0, 1.0, size=(N_B, N_W)) / np.sqrt(N_W)  # привод от мира

    def clone(self, **over):
        q = object.__new__(Params)
        q.__dict__.update(self.__dict__)
        for a, v in over.items():
            setattr(q, a, v)
        return q

    def label(self):
        return (f"k={self.k:g} c={self.c:g} w={self.w_drive:g} "
                f"nu={self.nu:g} eps={self.eps:g} seed={self.seed}")


def F_world(w, p):
    """Автономный предельный цикл. От выхода G не зависит по построению."""
    ct, st = np.cos(p.theta), np.sin(p.theta)
    rot = np.array([ct * w[0] - st * w[1], st * w[0] + ct * w[1]])
    r2 = float(w[0] * w[0] + w[1] * w[1])
    return rot * (1.0 + p.mu * (1.0 - r2))


def H(yhat, p):
    """Рулевой сигнал самомодели в тело. tanh ограничивает усиление."""
    return np.tanh(p.B @ yhat)


def build_input(w, w_prev, b, b_prev, x_g, p, rng):
    """Вход самомодели. Телеметрии нет: x_g попадает сюда только через c."""
    prop = p.c * (x_g + p.prop_noise * rng.normal(0, 1, size=N_G))
    return np.concatenate([w, w_prev, b, b_prev, prop])


class State:
    def __init__(self, p, rng):
        self.w = rng.normal(0, 0.5, size=N_W)
        self.w_prev = self.w.copy()
        self.b = rng.normal(0, 0.3, size=N_B)
        self.b_prev = self.b.copy()
        self.x_g = rng.normal(0, 0.1, size=N_G)
        self.x_gp = self.x_g.copy()
        self.x_dec = self.x_g.copy()

    def copy(self):
        s = object.__new__(State)
        for a in ("w", "w_prev", "b", "b_prev", "x_g", "x_gp", "x_dec"):
            setattr(s, a, getattr(self, a).copy())
        return s

    def target(self, branch):
        """Цель. Мир в неё не входит; размерность одинакова во всех ветках."""
        if branch == "yoked":
            return np.concatenate([self.b, self.x_gp])
        if branch == "null":
            return np.concatenate([self.b, self.x_dec])
        return np.concatenate([self.b, self.x_g])

    def op(self):
        """«Всё остальное» — знаменатель alpha."""
        return np.concatenate([self.w, self.b])


def step(state, p, G, Gp, rng, branch, dyhat=None, force_yhat=None):
    """Один такт. Возвращает (z_t, yhat_t, новое состояние).

    force_yhat подменяет выход модели заданным значением. Нужен для измерения
    путь-специфичного эффекта: возмущая «всё остальное», надо заблокировать
    косвенный путь через саму G, иначе знаменатель alpha раздувается тем же
    самовключением, которое alpha и должна измерять.
    """
    z = build_input(state.w, state.w_prev, state.b, state.b_prev, state.x_g, p, rng)
    yhat = G.predict(z) if force_yhat is None else force_yhat
    if dyhat is not None:
        # do(yhat): вмешательство в выходной регистр самомодели — переменную,
        # причинно входящую в цель, которую G предсказывает.
        yhat = yhat + dyhat
    yhat_p = Gp.predict(z) if Gp is not None else yhat

    new = State.__new__(State)
    new.w = F_world(state.w, p)
    new.w_prev = state.w.copy()
    new.b = (p.nu * state.b
             + p.w_drive * np.tanh(p.Wb @ state.w)
             + p.k * H(yhat, p)
             + p.noise * rng.normal(0, 1, size=N_B))
    new.b = np.clip(new.b, -6.0, 6.0)
    new.b_prev = state.b.copy()
    new.x_g = (1 - p.gamma) * state.x_g + p.gamma * np.tanh(p.A @ yhat)
    new.x_gp = (1 - p.gamma) * state.x_gp + p.gamma * np.tanh(p.A @ yhat_p)
    dec_drive = np.concatenate([state.b, state.x_dec])
    new.x_dec = (1 - p.gamma) * state.x_dec + p.gamma * np.tanh(p.A @ dec_drive)
    return z, yhat, new


def rollout(p, G, Gp, branch, T, seed_offset, burn_in=100):
    """Прогон замкнутого контура. Возвращает Z, Yhat, Y (цель на шаг вперёд)."""
    rng = np.random.default_rng(900_000 + 7919 * p.seed + seed_offset)
    st = State(p, rng)
    for _ in range(burn_in):
        _, _, st = step(st, p, G, Gp, rng, branch)
    Z = np.empty((T, DIM_Z))
    Yh = np.empty((T, N_Y))
    Y = np.empty((T, N_Y))
    for t in range(T):
        z, yhat, nxt = step(st, p, G, Gp, rng, branch)
        Z[t] = z
        # Положительный контроль: eps портит ТОЛЬКО отчётную копию предсказания.
        # Рулевая копия, ушедшая в step(), чистая, поэтому траектория при данном
        # seed побитово та же и R_ext не меняется.
        Yh[t] = yhat if p.eps == 0.0 else yhat + p.eps * rng.normal(0, 1, size=N_Y)
        Y[t] = nxt.target(branch)
        st = nxt
    return Z, Yh, Y
