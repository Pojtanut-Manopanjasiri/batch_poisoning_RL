# Batch Curation Poisoning — FVI with Per-Point Decomposition

Implementation of *Implementation Brief II: The Method* — fitted value iteration
(FVI) over a learned value function `V_hat(theta)`, a per-point / pairwise
decomposition of the action-value `Q`, three action-selection tiers behind one
interface, and the interaction diagnostics.

This is research/experiment code: simple, readable, no test suite.

## Note on Brief I

The brief assumes the environment/victim/baselines from *Brief I* already exist.
They were not present, so `env.py` reconstructs a **minimal** Brief I from the
problem recap: a linear-regression victim trained by online mini-batch gradient
descent (`l_i = 1/2 (<theta,x_i> - y_i)^2`). This is the natural minimal choice
that keeps the transition action-linear — the property the whole decomposition
rests on. Swapping in a different victim only requires a new `phi_batch`.

## Files

| file | contents |
|------|----------|
| `config.py`        | `EnvConfig`, `MethodConfig` (Brief II §6) |
| `env.py`           | `Stream`, `Victim`, `Environment`, feasibility helpers (Brief I) |
| `value_net.py`     | `ValueNet` — `V_hat(theta, t)`, terminal masking (§4.1) |
| `decomposition.py` | `scores_and_interaction` — alpha, B via autodiff (§3–4.2) |
| `selection.py`     | `select_action` — `sort` / `lowrank` / `exact` (§4.3) |
| `agents.py`        | baselines (null/random/greedy) + `MethodAgent` |
| `fvi.py`           | fitted value iteration loop + rollout/evaluate (§4.1) |
| `bdq.py`           | **BDQ baseline** — branching dueling Q-network (external comparison) |
| `diagnostics.py`   | `interaction_diagnostics` + the three figures (§5) |
| `viz.py`           | plots: training curve, cost-vs-budget, cost-vs-dimension (overall and by tier), duality gap, weight-decay sweep, per-policy action map, (d=1) regression-line attack, and (d>1) parameter-space attack |
| `run.py`           | end-to-end: train, compare to baselines, make figures |
| `experiment.ipynb` | notebook: trains, compares baselines, all diagnostics, and the d=1 regression-line demo |

## Run

```bash
python run.py                       # sort selection (default)
python run.py --mode lowrank --rank 2
python run.py --mode exact          # BQP; brute force here (no Gurobi installed)
python run.py --no-hessian          # Gram-only ablation for B
```

Outputs mean episode cost for null/random/greedy/method and writes
`fig1_spectrum.png`, `fig2_rel_interaction.png`, `fig3_tiers.png`.

## What to look for

- **Spectrum (fig 1).** With `d < m`, exactly `m - d` eigenvalues of `B` are ~0
  (rank(B) ≤ min(m, d)); the cumulative trace fraction shows how few directions
  carry the interaction — this motivates `lowrank`.
- **Relative interaction (fig 2).** `|½ aᵀBa| / |aᵀα|` grows ~linearly in `eta`.
- **Tier gap (fig 3).** `cost(exact) ≤ cost(lowrank) ≤ cost(sort)`. The gap is
  the price of ignoring the interaction — the headline result.

Caveat on the fig-3 wall-clock panel: at small `m` the brute-force `exact` path
uses a cached, vectorised enumerator and is very fast, so `lowrank`'s Python
dual-ascent loop can look slower. With Gurobi and larger `m`, `exact` becomes the
expensive tier.

## External baseline: BDQ (action branching)

`bdq.py` is a PyTorch port of **Branching Dueling Q-Network** (Tavakoli et al., AAAI
2018 — [action-branching-agents](https://github.com/atavakol/action-branching-agents),
MIT License). It is the natural model-free comparison: our joint action `{0,1,2}^m` has
`3^m` combinations (59049 at `m=10`), which plain DQN cannot enumerate; BDQ factors it
into `m` branches of 3 sub-actions, linear in `m`.

Ported faithfully from the reference's best settings: local-mean dueling aggregation,
`target_version='mean'`, `losses_version=2`, double-Q. Four adaptations are forced by
our problem (documented in the `bdq.py` docstring): cost- rather than reward-convention,
per-point conditioning via a permutation-equivariant encoder (weight-shared branches, so
the net is independent of `m`), feasibility projection reusing the *same* greedy-swap
rule as `selection._select_sort`, and `c_edit` reaching BDQ only through observed cost.

Result at `d=5, m=10, T=10, c_edit=0.1` (mean episode cost, 200 eval episodes):

| policy | cost | edits/step | training budget |
|---|---|---|---|
| null   | 16.43 | 0.00 | — |
| greedy | 16.00 | 0.31 | — |
| BDQ    | 11.42 | 1.11 | K=20 (same as method) |
| BDQ    |  9.89 | —    | K=50 (2.5× budget) |
| **method (FVI + sort)** | **8.55** | 4.06 | K=20 |

BDQ clearly beats the myopic greedy baseline — it does learn multi-step credit. But it
stays ~14% behind our method even at 2.5× the training budget, and it under-edits (1.1
vs 4.1 points/step). The likely cause is BDQ's known credit-assignment weakness: all `m`
branches regress on one shared scalar target, so an individual point's contribution is
diluted ~1/m by the other branches' choices. Our method instead gets each point's
contribution analytically as `alpha_i` from the action-linear transition.

## Dependencies

`torch`, `numpy`, `matplotlib`. `exact` uses Gurobi if `gurobipy` is installed,
otherwise brute force for `m ≤ 12` (with a `lowrank` fallback and a warning).
