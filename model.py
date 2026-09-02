"""Самомодель G_p: фиксированный базис + гребневой МНК. Никакого SGD.

    G_p(z) = W phi(z),   phi(z) = [1, z, cos(omega_1 z + b_1), ..., cos(omega_p z + b_p)]
    W = (Phi^T Phi + lambda I)^{-1} Phi^T Y

Ёмкость p — число случайных фурье-признаков; базис детерминирован по seed,
одинаков для всех веток внутри одной конфигурации. Веса находятся замкнутой
формулой, поэтому «схитрить» за счёт удачной инициализации или расписания
обучения система не может.
"""

import numpy as np

from dynamics import DIM_Z, N_B, N_G, N_Y

# Выходной регистр самомодели физически ограничен диапазоном самих целевых
# переменных: тело зажато в +-6, регистры в +-1 через tanh. Предсказывать
# недостижимое состояние регистр не может. Ограничение одинаково для всех веток
# и применяется только на выходе, не меняя замкнутую формулу МНК.
Y_BOUND = np.array([6.0] * N_B + [1.0] * N_G)


class Basis:
    """Случайные фурье-признаки. Детерминированы по (seed, p)."""

    def __init__(self, p, seed, lengthscale=1.5):
        rng = np.random.default_rng(500_000 + 31 * seed + p)
        self.p = int(p)
        self.omega = rng.normal(0, 1.0 / lengthscale, size=(self.p, DIM_Z))
        self.phase = rng.uniform(0, 2 * np.pi, size=self.p)
        self.n_feat = 1 + DIM_Z + self.p

    def __call__(self, Z):
        Z = np.atleast_2d(Z)
        rff = np.cos(Z @ self.omega.T + self.phase)
        return np.hstack([np.ones((Z.shape[0], 1)), Z, rff])


class Predictor:
    """Обученная (или нулевая) линейная модель над базисом."""

    def __init__(self, basis, W=None):
        self.basis = basis
        self.W = np.zeros((basis.n_feat, N_Y)) if W is None else W

    def predict(self, z):
        y = (self.basis(z) @ self.W)[0]
        return np.minimum(np.maximum(y, -Y_BOUND), Y_BOUND)

    def predict_batch(self, Z):
        return np.clip(self.basis(Z) @ self.W, -Y_BOUND, Y_BOUND)


def ridge(Phi, Y, lam=1e-6):
    n = Phi.shape[1]
    G = Phi.T @ Phi + lam * n * np.eye(n)
    return np.linalg.solve(G, Phi.T @ Y)


def fit_closed_loop(p_params, basis, branch, T, n_iter, lam=1e-6, twin=False):
    """Попеременная подгонка G внутри замкнутого контура.

    1. прогон с текущей G -> данные (z_t, y_{t+1})
    2. перефит W гребневым МНК
    3. повтор

    Данные, на которых учится G, порождены самой G. Это и есть требование
    неподвижной точки в функциональном пространстве. Сходимость итерации не
    гарантирована: при достаточно сильной обратной записи решения не существует,
    и невязка не падает. Несходимость здесь — ДАННЫЕ, а не сбой.

    Возвращает (G, G', история изменения весов).
    """
    from dynamics import rollout

    G = Predictor(basis)
    Gp = Predictor(Basis(basis.p, p_params.seed + 1)) if twin else None
    # Шумовой seed ФИКСИРОВАН по итерациям. Иначе изменение весов отражало бы
    # разброс выборки, а не сходимость к неподвижной точке: при k=0 статистика
    # держалась на 0.67 вместо нуля. Теперь между итерациями меняется только G.
    hist = []
    for it in range(n_iter):
        Z, _, Y = rollout(p_params, G, Gp, branch, T, seed_offset=0)
        W_new = ridge(basis(Z), Y, lam)
        hist.append(float(np.linalg.norm(W_new - G.W) / (np.linalg.norm(W_new) + 1e-12)))
        G = Predictor(basis, W_new)
        if twin:
            Zp, _, Yp = rollout(p_params, G, Gp, branch, T, seed_offset=100)
            Gp = Predictor(Gp.basis, ridge(Gp.basis(Zp), Yp, lam))
    return G, Gp, hist


def nrmse(Yhat, Y):
    """Нормированная невязка. 1.0 == не лучше предсказания среднего."""
    num = np.sqrt(np.mean(np.sum((Yhat - Y) ** 2, axis=1)))
    den = np.sqrt(np.mean(np.sum((Y - Y.mean(axis=0)) ** 2, axis=1)))
    return float(num / (den + 1e-12))
