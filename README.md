# Self-inclusion does not add prediction residual beyond loop closure

*[Читать по-русски](README.ru.md)* · **Repository:** https://github.com/cookiespooky/consciousness-revelation

A small, fully controlled experiment on closed-loop predictors. **Negative result.**

> In a closed-loop predictive system, the extra self-consistency residual comes from the **performativity of the loop** — the model's output changing the distribution it predicts — and not from whether the predictor includes its own physical realization in the set it models. Raising the measured self-inclusion α from 0.12 to 0.58 leaves the robust slope indistinguishable from zero at MDE = 0.0012, while a physically identical system *without* self-inclusion produces scatter of the same magnitude.

**This is not a claim about consciousness.** The word does not appear in any hypothesis, metric, or conclusion below. The project started from a philosophical intuition about self-representation; that intuition was reformulated into something measurable, and the measurable version did not survive.

---

## What was tested

A dynamical system split into three blocks:

| block | role |
|---|---|
| `w` (2 dims) | autonomous limit cycle. The model's output **cannot** influence it. Guarantees the task cannot be trivialized by freezing the world. Not part of the target. |
| `b` (3 dims) | body. Its causation is shared by three sources: own inertia `nu`, exogenous drive `w_drive` from the world, and the self-model's steering `k·H(ŷ)`. |
| `x_g` (2 dims) | the registers **in which the self-model is physically realized**. |

The self-model `G_p` is a fixed random-Fourier basis plus ridge least squares in closed form — no gradient training, no initialization luck, nothing to "cheat" with. It predicts `y = (b, x_g)` from `z = (w, w_prev, b, b_prev, c·x_g)`. **It receives no telemetry**: it learns about its own registers only through their consequences — the world loop (`k`) and a proprioceptive channel (`c`).

Because `y_{t+1}` depends on `ŷ_t`, the model must satisfy a functional fixed-point equation `G(z) = f(z) + k·H(G(z))`, which has no solution for strong enough coupling. Fitting is an alternating scheme: roll out with the current `G`, refit by ridge, repeat. **Non-convergence is data, not a bug.**

### Hypothesis

`self-inclusion → irreducible self-consistency residual`

Operationalized as: does the residual grow with the *measured* fraction of the target's causation that the model itself supplies?

### The two latent variables — measured, not set

Both are estimated **interventionally**, from a perturbation distribution fixed before any run and identical across every configuration:

```
alpha = influence of the model's output register on the TARGET
        ---------------------------------------------------------
        same + influence of everything else, WITH THE PATH THROUGH G BLOCKED

beta  = influence of the model's output register on the model's OWN INPUT
        / total influence on that input
```

The knobs are `k` (steering gain), `c` (proprioceptive gain), `w_drive` and `nu` (how much of the body's causation is exogenous). `alpha` and `beta` are then *measured*. `c` exists solely to move `beta` independently of `alpha`, so the primary coefficient is identifiable.

### Controls

| branch | what it isolates |
|---|---|
| `self` | model predicts `(b, x_g)` — its own registers are in the target |
| `yoked` | model predicts `(b, x_gp)` — a **twin's** registers. Same dimension, same dynamics, loop still closed, **no self-inclusion of the second block** |
| `external` | a passive observer of equal capacity predicts the same target on the same trajectory, but does not close the loop |
| `self2` | physically identical to `self`, differing **only in the random basis seed**. Gives the null level of any self-vs-X contrast |
| `null` | `k = 0` and a decoy target with no causal contact with the model. Proves the rig can output a zero |

Primary endpoint: **`S = Gap_self − Gap_yoked`**, where `Gap = R_branch − R_external`. Loop closure acts on both branches and cancels in the difference.

---

## Result

Final preregistered run: 30 seeds × 6 self-inclusion levels × 3 values of `c`, `k = 0.4` held fixed everywhere, capacity `p = 32`, 3600 rows. Primary estimator — **median (L1) regression**, chosen in advance precisely because ordinary least squares had proven sensitive to rare numerical blow-ups.

```
b_alpha (S)      = +0.00300   one-sided lower bound = -0.00112   CI [-0.00149, +0.00737]   n = 488
b_alpha (S_null) = -0.00036   one-sided lower bound = -0.00226
difference       = +0.00329   one-sided lower bound = -0.00102
matched pairs S  = +0.00332   one-sided lower bound = -0.00051
```

None of the three preregistered conditions is met. The secondary endpoint fails too.

**Sensitivity is not the limitation.** A fine calibration grid detected even the smallest injected perturbation: **MDE = 0.00121**. The effect across the whole α range is `0.0030 × 0.609 = 0.0018` — above MDE, but with a confidence interval covering zero.

**The effect shrank sixfold under a robust estimator** (`+0.0185 → +0.0030`). The earlier exploratory signal was in substantial part the heavy tail that ordinary least squares picks up and median regression does not.

### Supporting negative results

- **Loop closure explains almost everything.** In the first architecture, `Gap_yoked ≈ 0.88 × Gap_self` across the whole range.
- **Boundary of self-consistent model existence.** `k_c(self) = 1.041 [0.878, 1.208]`, `k_c(yoked) = 1.073 [0.908, 1.245]`, `Δk_c = −0.031 [−0.081, +0.015]`. The sign points the hypothesized way — but the `self`-vs-`self2` null gives `+0.038`, **larger in magnitude than the effect itself.**
- **Local dynamics.** `λ_max` (restricted to the model-driven subspace) coincides for `self` and `yoked` within SEM across the whole `k` range, and never crosses zero. So `k_c` marks the failure of the *functional* fixed point — no self-consistent predictor exists — and not dynamical instability: the state dynamics stay contracting throughout.

### Validity gate

The final run failed its own gate: `Gap(null) = +0.038` against a `0.02` tolerance, caused by three levels with `nu = 0.05` where the null branch (`k = 0`) has too little drive and its target collapses. Per-level *medians* there are `−0.0002 / −0.0001 / −0.0000`, so the zero reproduces perfectly by median and the mean is dragged by rare blow-ups.

Formally, no verdict is issued when the gate fails. On the subset where the null is alive (257 pairs, α from 0.115 to 0.423) the outcome is unchanged: `b_alpha(S) = +0.00334`, lower bound `−0.00056`; difference `+0.00313`, lower bound `−0.00104`. Conditions (1) and (3) fail there as well.

---

## The one lesson worth carrying elsewhere

**Compare against the scatter of your own procedure, not against zero.**

The `self2` control — the same physical system refit with a different random basis seed — twice produced differences of the same order as the effect being sought, and once larger than it. Without that control this project would have declared a result twice.

If you are measuring "self-modeling effects" in a learned system, build this control first. Everything else in the rig is negotiable; this is not.

## Traps the rig caught in itself

Each cost a run. Each generalizes to anyone building a self-modeling agent.

1. **Additive feedback destroys the attractor.** Trajectory std fell from 0.71 to 0.03 — the feedback simply clamps the system to a fixed point, and normalized residual starts measuring noise. Fix: the model's output modulates the dynamics instead of displacing it.
2. **The model learns to freeze the world.** Given steering over the phase, `G` finds a control that cancels rotation, making prediction trivial. Fix: the world became autonomous and unsteerable by construction.
3. **The intervention sat on the wrong variable.** Perturbing hidden registers `x_g` probes a path that does not exist at `c = 0`; α did not move with loop strength at all (0.209 → 0.185 → 0.064). Fix: intervene on `ŷ`, the variable that causally enters the target.
4. **α's denominator ran through the model itself.** Perturbing the world changed the model's input, and the "everything else" arm inflated with the very self-inclusion α is meant to measure. At `w_drive = 0` — pure self-loop — α read **0.02 instead of nearly 1**. Fix: path-specific effect; the model's output is held at its unperturbed value in that arm.
5. **Per-block σ scaling of the perturbation.** A weakly varying `ŷ` was perturbed weakly, understating the numerator — and the intervention distribution then differed per configuration, which the protocol forbade. Fix: one absolute magnitude for both arms.
6. **Removing the exogenous drive gives α ≈ 0.9 and nothing to measure.** The body loses every external source of variability, the target collapses to noise, all branches give `R ≈ 1.0`. High self-inclusion is reachable only at the price of a degenerate target.

Two more, in the diagnostics rather than the physics:

- **The convergence statistic measured sampling scatter.** Each fitting iteration used a fresh noise seed, so weight change never approached zero — 0.671 even at `k = 0`. With the noise seed fixed across iterations, only `G` changes: 0.012 at `k = 0`, rising monotonically to 4.45.
- **`λ_max` measured the wrong subsystem.** The Jacobian is block-triangular and the world is autonomous, so the top exponent belonged to its limit cycle and sat at −0.004 regardless of coupling. Fix: restrict the tangent vector to the model-driven subspace.

## Errors found in the analysis code itself

Listed because they would silently mislead anyone reusing this:

- MDE was computed in `Gap` units and compared against a regression *slope*.
- The collapse test called `+0.148` "small" against a base `R²` of `0.034` — a fivefold increase.
- Equivalence was declared from a small *point estimate* while the CI was 20× wider than the threshold. Equivalence requires the whole interval inside the bound.
- The positive control injected its perturbation through a path shared by both branches, so it cancelled in `S` and calibrated nothing.

---

## Running it

Requires `numpy` and `scipy`. `matplotlib` is optional — all analysis is computed from the raw CSV.

```bash
python3 run_final.py  raw_final.csv       # final preregistered run (~50 min, 3 cores)
python3 analysis_final.py raw_final.csv   # verdict by the preregistered criterion

python3 run.py smoke  raw_smoke.csv       # rig smoke test: null -> positive control -> branches
python3 run_stability.py raw_stability.csv         # k_c and lambda_max, self vs yoked
python3 run_stability.py raw_null.csv      self2   # the same contrast as a null
python3 analysis_stability.py raw_stability.csv raw_null.csv
```

Every run writes only raw rows. All figures and verdicts are derived from those files, so the analysis can be redone any number of times without touching the system.

| file | contents |
|---|---|
| `dynamics.py` | world / body / registers, feedback, rollout |
| `model.py` | random-Fourier basis, ridge, alternating closed-loop fit |
| `experiment.py` | branches, interventional α and β, residual, `ρ*`, degeneracy guard |
| `stability.py` | convergence criterion, `k_c` with bisection, `λ_max` by Benettin |
| `analysis*.py` | cluster bootstrap by seed, median regression, preregistered verdicts |
| `run_final.py` | **the preregistration is in this file's docstring** |
| `archive/` | the first architecture, kept for provenance of the v1 numbers |

## Limitations

Read these before citing anything above.

- One architecture. Fixed basis plus least squares, scalar steering into three body coordinates, `k` held at 0.4 in the final run.
- α was reached up to 0.93 with a converging fit, but the measurement range in the final run is 0.12–0.58. The `α → 1` limit is only reachable by removing the exogenous drive, which makes the target degenerate.
- The hypothesis was formulated by us. This is a negative result against our own operationalization of self-inclusion, not against a standing claim in the literature.
- `ρ*` (the coarseness a self-model needs to stay self-consistent) is collected but was never analyzed: its launch condition — confirmation of the primary effect — never arrived.

## Related work (positioning to verify)

What is measured here — a predictor whose output changes the distribution it predicts — appears to be the territory of **performative prediction** (Perdomo et al., 2020) and **auto-induced distributional shift**. The existence of a fixed point under increasing loop gain, which is what `k_c` measures, belongs to the same family. That framing both reduces the novelty of this work and predicts its result: the effect is a property of the loop, not of self-reference. This positioning has not been verified against the literature and should be checked before the work is presented anywhere.
