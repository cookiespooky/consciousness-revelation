# I tried hard to show self-reference is special. It isn't — my control was.

**Epistemic status:** A negative result from a toy model, with the hypothesis formulated by me rather than taken from the literature. I am confident in the null *within this system* and its measurement range, and genuinely uncertain how far it generalizes. The part I would defend most strongly is not the result but one control, which twice stopped me from announcing an effect that wasn't there. All code and raw data: https://github.com/cookiespooky/consciousness-revelation

---

## The intuition

Take a predictor embedded in the system it predicts, so that its output is written back into the state. Now ask whether it matters that the *variables the predictor physically runs on* are part of what it has to predict.

The intuition — mine, and I think a fairly common one — is that this is special. Predicting a thing you causally affect is one situation. Predicting a thing you causally affect *and which includes your own substrate* feels like it should be another, with an error floor that ordinary feedback doesn't impose. Something has to give when the map is part of the territory.

I built a small system to check, and spent a long series of runs trying to make the effect appear. It doesn't. What produces the residual is the loop being closed at all. Whether the loop passes through the model's own registers makes no measurable difference.

## Why I stopped saying "consciousness"

The project started from a philosophical conversation about self-representation — whether the strangeness of the first-person perspective might come from a system's model of itself necessarily being incomplete in a specific structural way.

That framing had to go before anything could be measured. Not because it's uninteresting, but because it makes the experiment unfalsifiable: any result can be narrated as supporting it. Dropping the word turned out to be the single most useful methodological decision. What remained was a claim with a number attached, and the number came out zero.

I recommend the move generally. If your hypothesis survives the removal of its most exciting word, it was a hypothesis.

## The setup, and the one control that mattered

Three blocks of state:

- a **world** that runs on a limit cycle and that the model provably cannot influence;
- a **body** whose motion is caused partly by the world and partly by the model's own steering, in a ratio I can dial;
- **registers** in which the self-model is physically realized.

The model predicts `(body, registers)` from what it can see. Crucially it gets **no telemetry** — it learns about its own registers only through their downstream effects. The model itself is a fixed random-feature basis with closed-form ridge regression: no gradient descent, no initialization luck, nothing to exploit.

I did not *set* how self-referential the system is. I exposed structural knobs and **measured** the resulting self-inclusion `α` interventionally: perturb the model's output register, see how much the prediction target moves, divide by that plus the effect of perturbing everything else — with the path through the model blocked, so the denominator doesn't inflate with the very thing I'm measuring.

Then the controls:

- **`yoked`** — the loop is closed exactly as before, the model still steers the body, but it predicts a *twin's* registers instead of its own. Same dimension, same dynamics, self-inclusion of that block removed. The primary quantity is `S = Gap_self − Gap_yoked`: loop closure hits both branches and cancels.
- **`null`** — no coupling at all, decoy target. Proves the rig can output a zero. (It does, to four decimal places.)
- **`self2`** — and this is the one. Physically the *same* system, refit with a different random basis seed. Nothing about it differs except a seed.

`self2` is the control I would keep if I had to throw away all the others.

## What happened

Preregistered final run, 30 seeds, median regression chosen in advance because ordinary least squares had already shown itself vulnerable to rare numerical blow-ups:

```
b_α (S)      = +0.00300   one-sided 95% lower bound = −0.00112
b_α (S_null) = −0.00036   one-sided 95% lower bound = −0.00226
difference   = +0.00329   one-sided 95% lower bound = −0.00102
matched pairs = +0.00332  one-sided 95% lower bound = −0.00051
```

Nothing crosses. Raising measured self-inclusion from 0.12 to 0.58 does not move the yoked-corrected residual.

**Sensitivity wasn't the problem.** I calibrated by injecting a known perturbation into only the *reported* copy of the prediction, leaving the steering copy clean so the trajectory stays bit-identical. Minimum detectable effect: **0.0012**. The effect I'm looking for, integrated over the whole α range, is 0.0018 — above the detection floor, with a confidence interval through zero. I could have seen it. It isn't there.

**The effect shrank sixfold when I switched to a robust estimator** — from +0.0185 to +0.0030. That difference is the heavy tail. Least squares was reporting it; median regression isn't.

Two supporting nulls, both of which I'd expected to be more interesting:

The **existence boundary** — the coupling strength past which no self-consistent predictor exists at all — differs between self and yoked by `Δk_c = −0.031`, in the hypothesized direction. Then I ran `self` against `self2`, and *that* contrast came out `+0.038`. **Bigger than the effect.** Changing a random seed moves the boundary more than removing self-reference does.

The **top Lyapunov exponent** of the model-driven subspace is identical for self and yoked across the entire coupling range, within standard error, and never crosses zero. Which also taught me something I hadn't distinguished: the fitting failure and dynamical instability are different thresholds. The state dynamics stay contracting the whole way. What breaks at `k_c` is the existence of a functional fixed point — there is no self-consistent predictor to converge to — not the stability of the system.

## The lesson

**Compare against the scatter of your own procedure, not against zero.**

Twice in this project the `self2` control was the only thing standing between me and a reported result. Both times the effect looked reasonable: right sign, plausible magnitude, monotone in the driving variable. Both times the same contrast between two fits of a *physically identical* system was as large or larger.

If you are measuring anything of the form "does self-modeling do X," build this control first. Not the elegant one, not the theoretically motivated one — the boring one where you change a seed and rerun. Everything else in my rig turned out to be negotiable. This didn't.

## Eight ways the apparatus lied to me

Each of these produced a wrong answer before I caught it. I'm listing them because I think most are generic to anyone building a self-modeling agent, and each cost a full run.

1. **Additive feedback destroyed the attractor.** Trajectory variance fell 20×; the feedback was just clamping the system to a fixed point, and my normalized residual started measuring noise instead of prediction.
2. **The model learned to freeze the world.** Given steering authority over phase, it found a control that cancels the rotation. Prediction becomes trivial when nothing moves. I had to make the world provably unsteerable.
3. **I intervened on the wrong variable.** Perturbing hidden registers probes a causal path that doesn't exist under some settings; my self-inclusion measure didn't respond to loop strength *at all* (0.209 → 0.185 → 0.064) and would have produced a confident false null.
4. **My denominator ran through the model.** Perturbing the world changed the model's input, which changed its output, which reached the target — so "everything else" inflated with exactly the self-inclusion I was trying to measure. In the pure self-loop limit this read **0.02 where the right answer is near 1**.
5. **Scaling perturbations per-block.** A weakly-varying output got weakly perturbed, understating the numerator — and the intervention distribution then differed per configuration, which my own protocol forbade.
6. **Removing the exogenous drive gets you α ≈ 0.9 and nothing to measure.** The target collapses to noise; every branch returns "no better than the mean." High self-inclusion is only reachable at the price of a degenerate target. I think that's a real constraint on this class of experiment, not just my bug.
7. **My convergence diagnostic measured sampling noise.** Fresh noise seed per fitting iteration meant weight change never went to zero — 0.671 even with the loop switched off. Fixing the seed across iterations: 0.012.
8. **My Lyapunov exponent measured the wrong subsystem.** Block-triangular Jacobian, autonomous world — the top exponent belonged to the world's limit cycle and sat at −0.004 no matter what I did to the coupling.

And four bugs in my own *analysis* code, which I mention because they're the kind that don't announce themselves: a detection threshold computed in one unit and compared against a regression slope in another; a model-comparison verdict that called a fivefold `R²` increase "small"; an equivalence claim made from a point estimate while the confidence interval was 20× wider than the threshold; and a positive control whose perturbation cancelled in the difference and therefore calibrated nothing while appearing to work.

The pattern I'd extract: **most of these failures produced clean, plausible, publishable-looking numbers.** None announced itself as a bug. They were caught by controls and sanity checks, not by inspection.

## What I believe now

In this system, the extra self-consistency error comes from the loop being performative — the model's output changing the distribution it predicts — and not from self-reference. The yoked control said so in the very first architecture (`Gap_yoked ≈ 0.88 × Gap_self`), and everything after that was me trying to find the missing 12% somewhere else and failing.

I suspect this is not new. The territory looks like **performative prediction** and **auto-induced distributional shift**, which already study predictors that change their own target distribution, including whether a fixed point exists as loop gain grows. If that framing is right, it predicts my result: the phenomenon belongs to the loop. I haven't verified the positioning carefully, and would welcome pointers.

## What would change my mind

- Reaching `α → 1` without the target degenerating. My range stops at 0.58 in the confirmatory design, and the top of the range is bought by removing exogenous drive, which kills the measurement. A design that escapes this trade-off would test the interesting limit.
- A learned rather than closed-form self-model. Everything here is ridge regression on a fixed basis. Gradient-trained models have their own dynamics, and the fixed-point structure might interact with them differently.
- A system where self-inclusion isn't confined to a small register block. Two registers out of a five-dimensional target is a modest amount of "self" to include.

If someone runs the `α → 1` version with a `self2`-style control and finds something, I'd update. If they run it without that control, I wouldn't.

---

**Code and raw data:** https://github.com/cookiespooky/consciousness-revelation

Every number in this post is derived from the raw CSVs in that repository, and the preregistration is recorded verbatim in the docstring of `run_final.py`, which is the script that produced the final dataset.
