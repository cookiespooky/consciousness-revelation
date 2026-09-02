"""Физика установки: состояние, оператор мира, обратная запись.

Состояние системы разбито на три блока фиксированной размерности:

    x_op  (n_op)  — операционный блок: (w1, w2) автономный предельный цикл,
                    НА КОТОРЫЙ ВЫХОД G НЕ ВЛИЯЕТ ВООБЩЕ, плюс телесная
                    координата b, которую G рулит с силой k. Мир нельзя
                    заморозить: иначе модель находит управление, останавливающее
                    систему, предсказание становится тривиальным и мерить нечего.
    x_g   (n_g)   — регистры, В КОТОРЫХ РЕАЛИЗОВАНА самомодель G
    x_gp  (n_g)   — регистры двойника G' (yoked-контроль), никогда не влияют на x_op
    x_dec (n_g)   — декоративный блок нулевой ветки: та же размерность и та же
                    форма динамики, но НУЛЕВОЙ причинный контакт с G

Порядок одного такта строго фиксирован:

    z_t      = вход самомодели (x_op[t], x_op[t-1], c * зашумлённый x_g[t])
    yhat_t   = G(z_t)              предсказание y_{t+1}
    yhat'_t  = G'(z_t)             двойник
    x_op[t+1] = F_op(x_op[t], steer=k*H(yhat_t)) + eta
    x_g[t+1]  = (1-gamma) x_g[t] + gamma * tanh(A yhat_t)
    x_gp[t+1] = (1-gamma) x_gp[t] + gamma * tanh(A yhat'_t)

Самовключение возникает потому, что y_{t+1} = (x_op[t+1], x_g[t+1]) зависит от
yhat_t. G обязана удовлетворять функциональному уравнению неподвижной точки

    G(z) = f(z) + k*H(G(z)) (+ соответствующий член по x_g),

решение которого существует не при всех k. Внешний наблюдатель O предсказывает
ту же величину, но не замыкает контур, и такого уравнения не решает.

G НЕ ПОЛУЧАЕТ ТЕЛЕМЕТРИЮ. Про собственные регистры она узнаёт только через
последствия: через мировой контур (сила k) и через проприоцептивный канал (сила c).
"""

import numpy as np

N_OP = 3          # (w1, w2) — автономный мир; b — телесная координата
N_G = 2
N_Y = N_OP + N_G          # размерность цели: (x_op, x_g) либо (x_op, x_gp)


class Params:
    """Все константы установки. Фиксируются до запуска и не трогаются."""

    def __init__(self, k=0.0, c=0.0, eps=0.0, seed=0,
                 theta=0.35, mu=0.30, gamma=0.40, nu=0.55,
                 noise=0.02, prop_noise=0.05, int_noise=0.0):
        self.k = float(k)                 # ручка: сила обратной записи в мир
        self.c = float(c)                 # ручка: усиление проприоцептивного канала
        self.eps = float(eps)             # положительный контроль, см. rollout()
        self.seed = int(seed)
        self.theta = theta                # угол поворота в F_op
        self.mu = mu                      # сила притяжения к предельному циклу
        self.gamma = gamma                # утечка регистров самомодели
        self.nu = nu                      # утечка телесной координаты
        self.noise = noise                # eta_t в мировой динамике
        self.prop_noise = prop_noise      # шум проприоцептивного канала
        self.int_noise = int_noise

        rng = np.random.default_rng(10_000 + self.seed)
        # H: R^N_Y -> R, рулевой канал в телесную координату.
        self.B = rng.normal(0, 1.0, size=(1, N_Y)) / np.sqrt(N_Y)
        # A: R^N_Y -> R^N_G, канал записи в собственные регистры.
        self.A = rng.normal(0, 1.0, size=(N_G, N_Y)) / np.sqrt(N_Y)

    def clone(self, **over):
        q = object.__new__(Params)
        q.__dict__.update(self.__dict__)
        for a, v in over.items():
            setattr(q, a, v)
        return q

    def label(self):
        return f"k={self.k:g} c={self.c:g} eps={self.eps:g} seed={self.seed}"


def F_op(x, p, steer=0.0):
    """Мир + тело.

    (w1, w2): поворот с притяжением к предельному циклу r=1. Автономны —
    выход самомодели на них не действует. Это гарантия того, что система не
    может быть остановлена: любая попытка G упростить себе задачу, заморозив
    мир, исключена конструктивно.

    b: телесная координата, протекающий интегратор, куда сходятся сигнал от мира
    и рулевой сигнал самомодели силой k. Именно здесь возникает самовключение:
    b[t+1] зависит от yhat_t, а b входит в цель, которую yhat_t предсказывает.
    """
    ct, st = np.cos(p.theta), np.sin(p.theta)
    w1 = ct * x[0] - st * x[1]
    w2 = st * x[0] + ct * x[1]
    r2 = float(x[0] * x[0] + x[1] * x[1])
    scale = 1.0 + p.mu * (1.0 - r2)
    b = p.nu * x[2] + np.tanh(1.3 * x[0] - 0.7 * x[1]) + steer
    return np.array([w1 * scale, w2 * scale, b])


def H(yhat, p):
    """Рулевой сигнал самомодели в телесную координату. tanh ограничивает усиление."""
    return float(np.tanh(p.B @ yhat)[0])


def build_input(x_op, x_op_prev, x_g, p, rng):
    """Вход самомодели z_t. Телеметрии нет: x_g попадает сюда только через c."""
    prop = p.c * (x_g + p.prop_noise * rng.normal(0, 1, size=N_G))
    return np.concatenate([x_op, x_op_prev, prop])


DIM_Z = N_OP * 2 + N_G


class State:
    """Полное физическое состояние установки."""

    def __init__(self, p, rng):
        self.x_op = rng.normal(0, 0.5, size=N_OP)
        self.x_op_prev = self.x_op.copy()
        self.x_g = rng.normal(0, 0.1, size=N_G)
        self.x_gp = self.x_g.copy()
        self.x_dec = self.x_g.copy()

    def copy(self):
        s = object.__new__(State)
        s.x_op = self.x_op.copy()
        s.x_op_prev = self.x_op_prev.copy()
        s.x_g = self.x_g.copy()
        s.x_gp = self.x_gp.copy()
        s.x_dec = self.x_dec.copy()
        return s

    def target(self, branch):
        """y_t для данной ветки. Размерность одинакова во всех ветках."""
        if branch == "yoked":
            return np.concatenate([self.x_op, self.x_gp])
        if branch == "null":
            return np.concatenate([self.x_op, self.x_dec])
        return np.concatenate([self.x_op, self.x_g])


def step(state, p, G, Gp, rng, branch, dyhat=None):
    """Один такт. Возвращает (z_t, yhat_t, новое состояние).

    G  — самомодель, замыкающая контур (её выход идёт в x_op и x_g).
    Gp — двойник; его выход идёт ТОЛЬКО в x_gp и никогда не влияет на мир.
    """
    z = build_input(state.x_op, state.x_op_prev, state.x_g, p, rng)
    yhat = G.predict(z)
    if dyhat is not None:
        # do(yhat): вмешательство в выходной регистр самомодели. Именно эта
        # переменная причинно входит в цель, которую G предсказывает, поэтому
        # именно она измеряет степень самовключения.
        yhat = yhat + dyhat
    yhat_p = Gp.predict(z) if Gp is not None else yhat

    eta = p.noise * rng.normal(0, 1, size=N_OP)
    new = State.__new__(State)
    new.x_op = F_op(state.x_op, p, steer=p.k * H(yhat, p)) + eta
    new.x_op = np.clip(new.x_op, -6.0, 6.0)
    new.x_op_prev = state.x_op.copy()
    new.x_g = (1 - p.gamma) * state.x_g + p.gamma * np.tanh(p.A @ yhat)
    new.x_gp = (1 - p.gamma) * state.x_gp + p.gamma * np.tanh(p.A @ yhat_p)
    # Декоративный блок: та же форма, но питается состоянием мира, а не выходом G.
    dec_drive = np.concatenate([state.x_op, state.x_dec])
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
        # Положительный контроль. eps портит ТОЛЬКО отчётную копию предсказания,
        # сравниваемую с целью. Рулевая копия, ушедшая в step(), осталась чистой,
        # поэтому траектория при данном seed побитово та же, R_ext не меняется, и
        # S растёт ровно на внесённую величину. Прошлая версия подмешивала помеху
        # во ВХОД G и тем самым ослабляла её рулевой сигнал — контроль получал
        # обратный знак и калибровал не ту величину.
        Yh[t] = yhat if p.eps == 0.0 else yhat + p.eps * rng.normal(0, 1, size=N_Y)
        Y[t] = nxt.target(branch)
        st = nxt
    return Z, Yh, Y
