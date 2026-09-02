# Loop closure, not self-reference: a controlled negative result on self-modeling predictors

**Categories:** cs.LG (primary), cs.AI

**Code and data:** https://github.com/cookiespooky/consciousness-revelation

---

## Abstract

A recurring intuition holds that a predictor which includes its own physical realization in the set it models should incur an irreducible self-consistency error, and that such self-reference is qualitatively different from ordinary feedback. We formalize this in a small, fully controlled dynamical system: an autonomous world the model cannot influence, a body whose causation is shared between exogenous drive and the model's own steering, and registers in which the self-model is physically realized. The self-model is a fixed random-feature basis fit by ridge regression in closed form and receives no telemetry about its own state; it learns about its registers only through their consequences. We measure self-inclusion α interventionally, as a path-specific causal fraction, rather than setting it by construction, and isolate it with a yoked control in which the loop is equally closed but a twin's registers replace the model's own in the prediction target. Under a preregistered protocol with median regression as the primary estimator, raising α from 0.12 to 0.58 leaves the slope of the yoked-corrected residual indistinguishable from zero (b_α = +0.0030, one-sided 95% lower bound −0.0011) at a calibrated minimum detectable effect of 0.0012. A physically identical system refit with a different random basis seed produces contrasts of the same magnitude, and in one case larger than the effect sought. The same null holds for the coupling strength at which a self-consistent predictor ceases to exist and for the top Lyapunov exponent of the model-driven subspace. We conclude that in this system the residual is a property of loop performativity rather than of self-reference. We additionally document eight failure modes of the apparatus, several of which produced apparent positive effects, and the same-system-different-seed control that eliminated them.

---

## 1. Introduction

Consider a predictor embedded in the system it predicts. Its output is written back into the state, so the quantity it must forecast depends on the forecast itself. A natural intuition, recurrent in discussions of self-modeling agents and of self-representation more generally, is that this situation is special when the predictor's *own physical substrate* is part of what it predicts — that self-reference of this kind should impose an error floor that ordinary output-to-state feedback does not.

This paper tests that intuition and reports that it does not hold in a system built specifically to give it every chance.

We separate two things that are easily conflated:

- **Loop closure (performativity).** The model's output changes the distribution it predicts. This holds whenever the prediction is written back, regardless of what is being predicted.
- **Self-inclusion.** The variables in which the model is physically realized are themselves members of the prediction target.

Our central control holds the first fixed and removes the second. If the intuition is right, the residual should grow with self-inclusion after loop closure has been subtracted. It does not.

We emphasize at the outset that the hypothesis tested here was formulated by us. This is a negative result against our own operationalization, not a refutation of a standing claim in the literature. We regard the transferable contribution as methodological: the control structure, the interventional measurement of self-inclusion, and a catalogue of failure modes that produced spurious positive results before being caught.

## 2. Related work

The phenomenon we isolate — a predictor whose output alters the distribution it predicts, and the question of whether a fixed point exists under increasing loop gain — appears to be the territory of **performative prediction** [Perdomo et al., 2020] and of **auto-induced distributional shift**. Work on self-modeling robots and on world models that include an agent's own body is adjacent but generally does not attempt to isolate self-inclusion from feedback.

This positioning has not been verified against the literature in detail. If it is correct, it both reduces the novelty of the present work and predicts its result: the residual should be a property of the loop, which is what we observe.

## 3. Setup

### 3.1 System

The state has three blocks:

- **World** `w ∈ ℝ²`: a rotation with attraction to a limit cycle at `r = 1`. The model's output cannot influence it, by construction. This is load-bearing: in an earlier version where the model could steer the world's phase, it learned a control that cancelled the rotation, freezing the system and making prediction trivial. The world is not part of the prediction target.
- **Body** `b ∈ ℝ³`: `b_{t+1} = ν·b_t + w_drive·tanh(W_b w_t) + k·tanh(B ŷ_t) + η_t`. Its causation is shared between inertia `ν`, exogenous drive `w_drive`, and the model's steering `k`.
- **Registers** `x_g ∈ ℝ²`: the variables in which the self-model is physically realized. `x_{g,t+1} = (1−γ)x_{g,t} + γ·tanh(A ŷ_t)`.

The prediction target is `y = (b, x_g)`. The input is `z = (w, w_prev, b, b_prev, c·x̃_g)`, where `x̃_g` is a noisy proprioceptive read scaled by `c`. **The model receives no telemetry**: it can learn about its own registers only through their consequences, via the world loop (gain `k`) and the proprioceptive channel (gain `c`). This mirrors the constraint that makes the problem non-trivial; with a privileged read of its own state, the phenomenon under study disappears by construction.

### 3.2 Self-model and fitting

`G_p(z) = W φ(z)` with `φ` a fixed random-Fourier basis of `p` features plus a linear and constant term, and `W` obtained by ridge regression in closed form. There is no gradient training, no initialization luck, and no learning-rate schedule — nothing the system can exploit to produce an artifact.

Because `y_{t+1}` depends on `ŷ_t`, the model must satisfy a functional fixed-point equation

```
G(z) = f(z) + k·H(G(z)) + …
```

which has no solution for sufficiently strong coupling. Fitting therefore proceeds by an alternating scheme: roll out with the current `G`, refit by ridge, repeat. Convergence of this scheme is a property of the problem, not of the optimizer, and non-convergence is treated as data. The noise seed is held fixed across iterations so that the only quantity changing between them is `G` itself.

## 4. Measured quantities

### 4.1 Self-inclusion, measured rather than set

The obvious way to vary self-inclusion is to choose which variables enter the target. That is a wiring decision: the experimenter picks the answer at design time. We instead expose structural knobs — `k`, `c`, `w_drive`, `ν` — and *measure* the resulting self-inclusion interventionally, using a perturbation distribution fixed before any run and identical across every configuration:

```
α = ‖Δ target under do(ŷ)‖ / ( ‖Δ target under do(ŷ)‖ + ‖Δ target under do(w, b) with ŷ held fixed‖ )
β = ‖Δ model input under do(ŷ)‖ / ( same + ‖Δ model input under do(w, b)‖ )
```

The denominator of α is **path-specific**: when perturbing the world and body, the model's output is clamped to its unperturbed value. Without this, the perturbation reaches the target *through the model*, and the denominator inflates with the very self-inclusion α is meant to measure — in the pure self-loop limit this produced α ≈ 0.02 where the correct value is near 1.

β exists because `k` moves self-inclusion and self-observability together. Holding it as a covariate, with `c` varied to move it independently of α, is what makes the primary coefficient identifiable; a design without this yields `|corr(α, β)| ≈ 0.9` and an uninterpretable estimate.

### 4.2 Branches

| branch | isolates |
|---|---|
| `self` | target `(b, x_g)` — the model's own registers are included |
| `yoked` | target `(b, x_g′)` — a twin's registers. Same dimension, same dynamics, loop equally closed, self-inclusion of the second block removed |
| `external` | a passive observer of equal capacity predicts the same target on the same trajectory without closing the loop |
| `self2` | physically identical to `self`, differing only in the random basis seed |
| `null` | `k = 0` and a decoy target with no causal contact with the model |

With `Gap = R_branch − R_external` and `R` a normalized RMSE, the primary quantity is

```
S = Gap_self − Gap_yoked
```

Loop closure acts on both branches and cancels. `S_null = Gap_self − Gap_self2` gives the scatter the fitting procedure itself produces.

## 5. Preregistration

Fixed before the final run:

- **Primary model:** median (L1) regression `S = b₀ + b_α α + b_β β`, on non-degenerate self/yoked pairs at `p = 32`. Median regression was chosen in advance because ordinary least squares had already proven sensitive to rare numerical blow-ups; winsorization was considered and rejected.
- **Hypotheses:** `H₀: b_α ≤ 0` against `H₁: b_α > 0`.
- **Detection requires all three:** (1) the one-sided 95% lower bound (5th percentile of a cluster bootstrap over seeds) for `b_α(S)` exceeds zero; (2) the same criterion fails for `b_α(S_null)`; (3) the lower bound on the difference of slopes exceeds zero.
- **Exclusion rule:** a configuration is dropped if `max(conv_self, conv_yoked)` exceeds `τ = 0.25`, a convergence threshold introduced earlier for an unrelated purpose and not tuned here, or if target degeneracy is flagged.
- **Secondary:** matched pairs — the slope in α computed within each seed, then aggregated. Does not replace the primary test under any outcome.
- **Validity gates:** `|corr(α, β)| < 0.85`; `|Gap(null)| < 0.02`; MDE established by a fine ε grid rather than by its lower endpoint.

## 6. Results

Final run: 30 seeds × 6 self-inclusion levels × 3 values of `c`, with `k = 0.4` held fixed throughout so that only the exogenous share of the body's causation varies. 3600 rows.

```
b_α (S)      = +0.00300   one-sided lower bound = −0.00112   CI [−0.00149, +0.00737]   n = 488
b_α (S_null) = −0.00036   one-sided lower bound = −0.00226
difference   = +0.00329   one-sided lower bound = −0.00102
matched pairs = +0.00332  one-sided lower bound = −0.00051
```

None of the three conditions is met; the secondary endpoint fails as well.

**Sensitivity is not the binding constraint.** A calibration in which a known perturbation of size ε corrupts only the reported copy of the prediction — leaving the steering copy clean, so the trajectory is bit-identical and `R_external` is unchanged — detected even the smallest injected value. **MDE = 0.00121.** The effect across the full α range is `0.0030 × 0.609 = 0.0018`, above MDE, but with a confidence interval covering zero.

**The effect shrank sixfold under the robust estimator.** An exploratory run had given an OLS slope of `+0.0185`; median regression on the confirmatory design gives `+0.0030`. The earlier signal was in substantial part the heavy tail that least squares absorbs.

### 6.1 Loop closure accounts for nearly all of the residual

In the first architecture, `Gap_yoked ≈ 0.88 × Gap_self` across the entire range. The yoked control was designed to test whether self-inclusion adds anything to loop closure; its answer was that loop closure is almost the whole effect.

### 6.2 Existence boundary of a self-consistent model

Defining `k_c` as the largest coupling at which the alternating fit converges, refined by bisection:

```
k_c(self)  = 1.041   95% CI [0.878, 1.208]
k_c(yoked) = 1.073   95% CI [0.908, 1.245]
Δk_c       = −0.031  95% CI [−0.081, +0.015]
```

The sign points the hypothesized way. However, the same contrast computed between `self` and `self2` — physically identical systems differing only in basis seed — gives `+0.038`, **larger in magnitude than the effect**. The measured shift lies entirely inside the scatter produced by changing a random seed.

### 6.3 Local dynamics

The top Lyapunov exponent, restricted to the model-driven subspace, coincides for `self` and `yoked` within standard error across the whole range of `k` (−0.507 to −0.40) and never crosses zero. Two consequences: there is no self-inclusion effect in the local dynamics, and `k_c` marks the failure of the *functional* fixed point — no self-consistent predictor exists — rather than dynamical instability, since the state dynamics remain contracting throughout.

### 6.4 Validity gate

The final run failed its own gate: `Gap(null) = +0.038` against a tolerance of `0.02`, caused by three levels at which the null branch (`k = 0`) has too little drive and its target collapses; per-level medians there are `−0.0002`, `−0.0001`, `−0.0000`, so the zero reproduces by median and the mean is dragged by rare blow-ups.

Formally no verdict is issued when a gate fails. On the subset where the null is intact (257 pairs, α from 0.115 to 0.423) the outcome is unchanged: `b_α(S) = +0.00334`, lower bound `−0.00056`; difference `+0.00313`, lower bound `−0.00104`.

## 7. Methodological findings

### 7.1 The control that mattered

**Compare against the scatter of the fitting procedure, not against zero.** The `self2` control — the same physical system refit with a different random basis seed — twice produced differences of the same order as the effect being sought, and once larger. Without it this study would have reported a positive result twice: once for `Δk_c` and once for the exploratory slope.

We recommend that any measurement of "self-modeling effects" in a learned system include this control before any other.

### 7.2 Failure modes of the apparatus

Each of the following produced a wrong answer before being caught, and each generalizes beyond this particular system.

1. **Additive feedback destroys the attractor.** Trajectory standard deviation fell from 0.71 to 0.03: the feedback clamps the system to a fixed point and normalized residual begins measuring noise. Remedy: the output modulates the dynamics rather than displacing the state.
2. **The model learns to freeze the world.** Given steering authority over phase, it finds a control cancelling the rotation, making prediction trivial. Remedy: an autonomous, unsteerable world.
3. **The intervention on the wrong variable.** Perturbing hidden registers probes a path that does not exist when `c = 0`; α failed to move with loop strength at all (0.209 → 0.185 → 0.064). Remedy: intervene on the output register, the variable that causally enters the target.
4. **A denominator that runs through the model.** Described in §4.1; produced α ≈ 0.02 in the pure self-loop limit.
5. **Per-block σ scaling of the perturbation.** A weakly varying output was perturbed weakly, understating the numerator, and the intervention distribution then differed across configurations, which the protocol forbade. Remedy: one absolute magnitude for both arms.
6. **Removing the exogenous drive yields α ≈ 0.9 and nothing to measure.** The target collapses to noise and every branch returns `R ≈ 1.0`. High self-inclusion is reachable only at the cost of a degenerate target — itself a substantive constraint on this class of experiment.
7. **A convergence statistic measuring sampling scatter.** With a fresh noise seed per fitting iteration, weight change never approached zero (0.671 even at `k = 0`). With the seed fixed, 0.012 at `k = 0`, rising monotonically to 4.45.
8. **A Lyapunov exponent of the wrong subsystem.** The Jacobian is block-triangular and the world is autonomous, so the top exponent belonged to its limit cycle and sat at −0.004 irrespective of coupling. Remedy: restrict the tangent vector to the model-driven subspace.

### 7.3 Errors in the analysis code

Reported because they would silently mislead reuse: a minimum detectable effect computed in units of the dependent variable and compared against a regression *slope*; a model-comparison verdict calling a fivefold increase in `R²` negligible; equivalence declared from a small point estimate while the confidence interval was twenty times wider than the threshold; and a positive control whose perturbation travelled a path shared by both branches, cancelling in the difference and calibrating nothing.

## 8. Limitations

- A single architecture: fixed random features with ridge regression, scalar steering into three body coordinates, `k` fixed at 0.4 in the confirmatory run.
- α reached 0.93 with a converging fit, but the measurement range in the final design is 0.12–0.58. The `α → 1` limit is attainable only by removing the exogenous drive, which degenerates the target (§7.2, item 6).
- The hypothesis was formulated by the authors. This is a negative result against a specific operationalization of self-inclusion.
- One secondary quantity — the coarseness a self-model requires to remain self-consistent — was collected but not analyzed, as its preregistered launch condition (confirmation of the primary effect) was not met.

## 9. Conclusion

In a closed-loop predictive system built to make self-reference maximally visible, the additional self-consistency residual is a property of the performativity of the loop and not of whether the predictor includes its own physical realization in the set it models. Raising measured self-inclusion from 0.12 to 0.58 leaves the robust slope indistinguishable from zero at a calibrated sensitivity of 0.0012, and a physically identical system refit with a different random seed produces scatter of the same magnitude. The same null holds at the existence boundary of a self-consistent model and in the local dynamics.

We do not claim this settles the general question. We claim that the specific and appealing form of the intuition — that self-reference *per se* imposes an error floor — is measurable, was measured, and did not survive a design with the controls required to distinguish it from ordinary feedback and from the noise of the fitting procedure itself.

## Availability

Code and all raw result files are at https://github.com/cookiespooky/consciousness-revelation ; every figure and verdict in this note is derived from the raw CSVs, and the preregistration is recorded verbatim in the docstring of the run script that produced the final data.
