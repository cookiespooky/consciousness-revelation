# Archive

`dynamics_v1.py` — the first architecture, in which the autonomous world was part of the
prediction target. It produced the result cited in the main README as
`Gap_yoked ≈ 0.88 × Gap_self`, along with `raw_smoke.csv` and `raw_probe.csv`.

It is kept for provenance only and **will not run against the current modules**: the state
layout changed when the world was moved out of the target in order to lift the structural
ceiling on measured self-inclusion (α could not exceed ~0.25 while the autonomous world
dominated the denominator).
