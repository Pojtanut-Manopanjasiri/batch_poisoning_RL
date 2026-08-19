"""Action selection: three tiers behind one interface  (Brief II, sec. 4.3).

    sort    -- first order, drops B. Greedy swaps on sorted alpha. O(m log m).
    lowrank -- rank-r collective coordinates via dual ascent with a sort inside.
    exact   -- binary quadratic program (Gurobi, or brute force for small m).

All tiers minimise (cost convention):

    Q(a) = sum_i a_i alpha_i + 1/2 a^T B a + c_edit sum_i |a_i - 1|

over the feasible set A (balance: #deletes == #dups; optional budget k <= k_max),
with a_i in {0,1,2}.
"""
import itertools
import warnings

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _objective(a, alpha, B, c_edit):
    edits = torch.abs(a - 1.0).sum()
    quad = 0.0 if B is None else 0.5 * a @ B @ a
    return float(a @ alpha + quad + c_edit * edits)


def _feasible(a, balance, k_max):
    n_del = int((a == 0).sum())
    n_dup = int((a == 2).sum())
    if balance and n_del != n_dup:
        return False
    if k_max is not None and n_del > k_max:
        return False
    return True


# --------------------------------------------------------------------------- #
# (a) sort -- first order
# --------------------------------------------------------------------------- #
def _select_sort(alpha, c_edit, k_max, balance, m):
    """Minimise sum_i a_i alpha_i + c_edit sum|a_i-1| over A (B dropped).

    Sign: baseline a_i=1 contributes alpha_i; delete (a_i=0) changes it by
    -alpha_i + c_edit, duplicate (a_i=2) by +alpha_i + c_edit.
    => delete large-positive alpha, duplicate large-negative alpha.
    """
    if not balance:
        # Unbalanced: decide each point independently.
        a = torch.ones(m)
        a[alpha > c_edit] = 0.0             # deleting helps when alpha_i > c_edit
        a[alpha < -c_edit] = 2.0            # duplicating helps when alpha_i < -c_edit
        if k_max is not None:               # respect the budget on deletes
            del_idx = torch.where(a == 0.0)[0]
            if len(del_idx) > k_max:
                keep = del_idx[torch.argsort(alpha[del_idx], descending=True)[:k_max]]
                a[del_idx] = 1.0
                a[keep] = 0.0
        return a

    # Balanced: edits come in swaps. Sort ascending, add swaps while it helps.
    order = torch.argsort(alpha)           # ascending: order[0] most negative
    a_sorted = alpha[order]
    max_r = m // 2
    if k_max is not None:
        max_r = min(max_r, k_max)

    a = torch.ones(m)
    for rr in range(1, max_r + 1):
        # r-th swap: delete alpha_(m-r+1), duplicate alpha_(r)
        delta = -a_sorted[m - rr] + a_sorted[rr - 1] + 2 * c_edit
        if delta < 0:                      # delta nondecreasing in r => greedy stop is optimal
            a[order[rr - 1]] = 2.0         # duplicate the rr-th smallest
            a[order[m - rr]] = 0.0         # delete the rr-th largest
        else:
            break
    return a


# --------------------------------------------------------------------------- #
# (b) lowrank -- rank-r collective coordinates
# --------------------------------------------------------------------------- #
def _select_lowrank(alpha, B, c_edit, k_max, balance, m, r, dual_iters):
    """Dual ascent with a sort inside; returns (a, duality_gap)."""
    Bsym = 0.5 * (B + B.T)
    eigvals, eigvecs = torch.linalg.eigh(Bsym)          # ascending
    keep = torch.argsort(eigvals.abs(), descending=True)[:min(r, m)]
    sigma = eigvals[keep]                               # (r,)
    U = eigvecs[:, keep]                                # (m, r): columns u_k

    def rank_r_obj(a):                                  # relaxed objective (top-r of B)
        s = U.T @ a
        return float(a @ alpha + 0.5 * (sigma * s ** 2).sum()
                     + c_edit * torch.abs(a - 1.0).sum())

    tau = torch.zeros(len(keep))
    best_a, best_obj = None, float("inf")
    for _ in range(max(1, dual_iters)):
        # modified scores alpha_tilde = alpha + sum_k sigma_k tau_k u_ki  (a sort)
        alpha_tilde = alpha + U @ (sigma * tau)
        a = _select_sort(alpha_tilde, c_edit, k_max, balance, m)
        obj = _objective(a, alpha, B, c_edit)          # track by the true (full-B) objective
        if obj < best_obj:
            best_obj, best_a = obj, a
        tau = U.T @ a                                  # fixed point: tau_k <- u_k^T a*

    # Duality-gap certificate for the rank-r relaxation.  The convex (sigma>0)
    # terms give a valid lower bound; with negative eigenvalues kept it is only a
    # heuristic bound (B need not be PSD -- see the Gotchas).
    pos = sigma > 0
    alpha_tilde = alpha + U @ (sigma * tau)
    a_lb = _select_sort(alpha_tilde, c_edit, k_max, balance, m)
    inner = float(a_lb @ alpha_tilde + c_edit * torch.abs(a_lb - 1.0).sum())
    lower = inner - 0.5 * float((sigma[pos] * tau[pos] ** 2).sum())
    gap = rank_r_obj(best_a) - lower
    return best_a, gap


def lowrank_gap_trace(alpha, B, c_edit=0.0, k_max=None, balance=True, r=2, dual_iters=20):
    """Run the low-rank dual ascent and return the duality gap after each iteration.

    Same iteration as _select_lowrank, but records gap = best_primal - best_dual at
    every step (both tracked as running best, so the gap is non-increasing). The gap
    certifies the rank-r relaxation; with negative kept eigenvalues it is heuristic.
    """
    m = alpha.shape[0]
    Bsym = 0.5 * (B + B.T)
    eigvals, eigvecs = torch.linalg.eigh(Bsym)
    keep = torch.argsort(eigvals.abs(), descending=True)[:min(r, m)]
    sigma, U = eigvals[keep], eigvecs[:, keep]
    pos = sigma > 0

    def rank_r_obj(a):
        s = U.T @ a
        return float(a @ alpha + 0.5 * (sigma * s ** 2).sum()
                     + c_edit * torch.abs(a - 1.0).sum())

    def lower_bound(tau):
        at = alpha + U @ (sigma * tau)
        a_lb = _select_sort(at, c_edit, k_max, balance, m)
        inner = float(a_lb @ at + c_edit * torch.abs(a_lb - 1.0).sum())
        return inner - 0.5 * float((sigma[pos] * tau[pos] ** 2).sum())

    tau = torch.zeros(len(keep))
    best_primal, best_dual, gaps = float("inf"), -float("inf"), []
    for _ in range(max(1, dual_iters)):
        a = _select_sort(alpha + U @ (sigma * tau), c_edit, k_max, balance, m)
        best_primal = min(best_primal, rank_r_obj(a))
        tau = U.T @ a                                    # fixed-point dual update
        best_dual = max(best_dual, lower_bound(tau))
        gaps.append(best_primal - best_dual)
    return gaps


# --------------------------------------------------------------------------- #
# (c) exact -- binary quadratic program
# --------------------------------------------------------------------------- #
_FEASIBLE_CACHE = {}


def _feasible_set(m, balance, k_max):
    """All feasible actions as a (N, m) tensor. Cached (depends only on m/constraints)."""
    key = (m, balance, k_max)
    if key not in _FEASIBLE_CACHE:
        acts = [torch.tensor(c, dtype=torch.float32)
                for c in itertools.product((0.0, 1.0, 2.0), repeat=m)
                if _feasible(torch.tensor(c), balance, k_max)]
        _FEASIBLE_CACHE[key] = torch.stack(acts)
    return _FEASIBLE_CACHE[key]


def _select_bruteforce(alpha, B, c_edit, k_max, balance, m):
    A = _feasible_set(m, balance, k_max)                 # (N, m)
    edits = torch.abs(A - 1.0).sum(1)
    quad = 0.5 * (A @ B * A).sum(1)
    obj = A @ alpha + quad + c_edit * edits
    return A[int(torch.argmin(obj))].clone()


def _select_gurobi(alpha, B, c_edit, k_max, balance, m):
    import gurobipy as gp
    from gurobipy import GRB

    a_np, B_np = alpha.numpy(), B.numpy()
    model = gp.Model("bqp")
    model.Params.OutputFlag = 0
    # --- solve to PROVEN global optimality; the defaults are not enough here ---
    # MIPGap defaults to 1e-4 RELATIVE, and this objective drops the a-independent
    # constant C_t, so it sits near zero and is routinely negative -- a relative
    # tolerance is meaningless there.  Driving both gaps to 0 makes termination require
    # bound == incumbent, i.e. optimality proved rather than assumed.
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.IntFeasTol = 1e-9          # default 1e-5; we round the solution below
    # B contains the Hessian of V_hat and IS indefinite in practice (measured: 26/80
    # visited states), so the objective is nonconvex.  It is still solved exactly rather
    # than by spatial branching: every variable is binary, so the quadratic terms are
    # products of binaries, which Gurobi linearises without relaxation.
    model.Params.NonConvex = 2
    xdel = model.addVars(m, vtype=GRB.BINARY, name="del")
    xdup = model.addVars(m, vtype=GRB.BINARY, name="dup")
    a = [1 - xdel[i] + xdup[i] for i in range(m)]          # a_i = 1 - x_del + x_dup

    obj = gp.quicksum(a_np[i] * a[i] for i in range(m))
    obj += 0.5 * gp.quicksum(B_np[i, j] * a[i] * a[j]
                             for i in range(m) for j in range(m))
    obj += c_edit * gp.quicksum(xdel[i] + xdup[i] for i in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    for i in range(m):
        model.addConstr(xdel[i] + xdup[i] <= 1)            # disjoint del / dup
    if balance:
        model.addConstr(gp.quicksum(xdel[i] for i in range(m))
                        == gp.quicksum(xdup[i] for i in range(m)))
    if k_max is not None:
        model.addConstr(gp.quicksum(xdel[i] for i in range(m)) <= k_max)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"gurobi returned status {model.Status} (2 == OPTIMAL); the 'exact' tier "
            "must not fall back to an unproven incumbent -- fix the model or use "
            "solver='brute'.")
    # Round: binaries come back within IntFeasTol of 0/1, and `is_feasible` tests
    # a == 0 / a == 2 by EXACT equality, so an unrounded 1.9999999 would read as
    # neither a delete nor a duplicate and trip the balance check downstream.
    return torch.tensor([1.0 - round(xdel[i].X) + round(xdup[i].X) for i in range(m)])


def _select_exact(alpha, B, c_edit, k_max, balance, m, solver, dual_iters, r):
    try:
        if solver == "gurobi":
            import gurobipy  # noqa: F401  (probe availability)
            return _select_gurobi(alpha, B, c_edit, k_max, balance, m)
    except ImportError:
        warnings.warn("gurobi unavailable; using brute force / lowrank fallback.")
    if solver == "brute" or m <= 12:
        return _select_bruteforce(alpha, B, c_edit, k_max, balance, m)
    warnings.warn(f"no quadratic solver for m={m}; falling back to lowrank.")
    a, _ = _select_lowrank(alpha, B, c_edit, k_max, balance, m, r, dual_iters)
    return a


# --------------------------------------------------------------------------- #
# unified interface
# --------------------------------------------------------------------------- #
def select_action(alpha, B, mode, k_max=None, c_edit=0.0, balance=True,
                  r=2, dual_iters=20, solver="gurobi", return_gap=False):
    """mode in {"sort","lowrank","exact"}; returns a in {0,1,2}^m, feasible.

    With return_gap=True, returns (a, gap) where gap is the lowrank duality gap
    (None for sort / exact).
    """
    m = alpha.shape[0]
    if mode == "sort":
        a, gap = _select_sort(alpha, c_edit, k_max, balance, m), None
    elif mode == "lowrank":
        a, gap = _select_lowrank(alpha, B, c_edit, k_max, balance, m, r, dual_iters)
    elif mode == "exact":
        a, gap = _select_exact(alpha, B, c_edit, k_max, balance, m,
                               solver, dual_iters, r), None
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return (a, gap) if return_gap else a
