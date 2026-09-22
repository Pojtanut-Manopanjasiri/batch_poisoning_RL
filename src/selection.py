"""Action selection: two tiers behind one interface  (Brief II, sec. 4.3).

    greedy  -- Algorithm 2: first order, drops Gamma.  Exact for the surrogate.
    exact   -- integer quadratic program (Gurobi, or brute force for small m).

Both minimise (cost convention)

    Q(a) = sum_i a_i alpha_i + 1/2 a^T Gamma a + c_edit ||a - 1||_1

over A = {a in {0,...,L}^m : sum_i a_i = m, ||a - 1||_1 <= B}.
"""
import itertools
import warnings

import torch


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _objective(a, alpha, Gamma, c_edit):
    edits = torch.abs(a - 1.0).sum()
    quad = 0.0 if Gamma is None else 0.5 * a @ Gamma @ a
    return float(a @ alpha + quad + c_edit * edits)


def _feasible(a, balance, B, L):
    if bool((a < 0).any()) or bool((a > L).any()):
        return False
    if balance and int(a.sum()) != len(a):
        return False
    if B is not None and float((a - 1.0).abs().sum()) > B:
        return False
    return True


# --------------------------------------------------------------------------- #
# (a) greedy -- Algorithm 2, first order (drops Gamma)
# --------------------------------------------------------------------------- #
def _select_greedy(alpha, c_edit, B, balance, m, L):
    """Minimise sum_i alpha_i a_i + c_edit ||a-1||_1 over A  (Algorithm 2).

    Marginal cost of raising a_i from s: d_i(s) = alpha_i - rho if s == 0 else
    alpha_i + rho.  Convex in a_i, so greedy allocation is exact.
    """
    rho = c_edit
    if not balance:
        # each point independently: 0 if alpha_i > rho, L if alpha_i < -rho, else 1
        a = torch.ones(m)
        a[alpha > rho] = 0.0
        a[alpha < -rho] = float(L)
        if B is not None:
            cost = (a - 1.0).abs()                       # budget each point spends
            gain = alpha * 1.0 - alpha * a               # cost(keep) - cost(chosen)
            gain = gain + rho * (1.0 - cost)
            spent, order = 0.0, torch.argsort(gain, descending=True)
            keep = torch.ones(m)
            for i in order:
                if spent + float(cost[i]) <= B:
                    keep[i] = a[i]
                    spent += float(cost[i])
            a = keep
        return a

    k = m if B is None else B // 2       # under sum a = m, ||a-1||_1 = 2 * #dups
    a = torch.zeros(m)
    D = 0                                # duplication increments used so far
    for _ in range(m):                   # exactly m allocations => sum_i a_i = m
        d = alpha + torch.where(a == 0, -rho, rho)
        elig = (a < L) & ((a == 0) | (D < k))
        d = d.masked_fill(~elig, float("inf"))
        i = int(torch.argmin(d))
        if a[i] >= 1:
            D += 1
        a[i] += 1
    return a


# --------------------------------------------------------------------------- #
# (b) exact -- integer quadratic program
# --------------------------------------------------------------------------- #
_FEASIBLE_CACHE = {}


def _feasible_set(m, balance, B, L):
    """All feasible actions as a (N, m) tensor. Cached; (L+1)^m before filtering."""
    key = (m, balance, B, L)
    if key not in _FEASIBLE_CACHE:
        acts = [torch.tensor(c, dtype=torch.float32)
                for c in itertools.product(*[range(L + 1)] * m)
                if _feasible(torch.tensor(c, dtype=torch.float32), balance, B, L)]
        _FEASIBLE_CACHE[key] = torch.stack(acts)
    return _FEASIBLE_CACHE[key]


def _select_bruteforce(alpha, Gamma, c_edit, B, balance, m, L):
    A = _feasible_set(m, balance, B, L)
    edits = torch.abs(A - 1.0).sum(1)
    quad = 0.5 * (A @ Gamma * A).sum(1)
    obj = A @ alpha + quad + c_edit * edits
    return A[int(torch.argmin(obj))].clone()


def _select_gurobi(alpha, Gamma, c_edit, B, balance, m, L):
    import gurobipy as gp
    from gurobipy import GRB

    a_np, G_np = alpha.numpy(), Gamma.numpy()
    model = gp.Model("iqp")
    model.Params.OutputFlag = 0
    # MIPGap defaults to 1e-4 RELATIVE, and this objective drops the a-independent
    # constant C_t, so it sits near zero and is routinely negative -- a relative
    # tolerance is meaningless there.  Both gaps 0 => optimality proved.
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.IntFeasTol = 1e-9
    # Gamma carries the Hessian of h and IS indefinite in practice, so the objective
    # is nonconvex.  With L > 1 the variables are general integers rather than
    # binaries, so Gurobi cannot linearise the products and falls back to spatial
    # branching -- correct, but slower than the L = 1 binary case.
    model.Params.NonConvex = 2
    a = model.addVars(m, vtype=GRB.INTEGER, lb=0, ub=L, name="a")
    dev = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="dev")   # |a_i - 1|
    for i in range(m):
        model.addConstr(dev[i] >= a[i] - 1)
        model.addConstr(dev[i] >= 1 - a[i])
    if balance:
        model.addConstr(gp.quicksum(a[i] for i in range(m)) == m)
    if B is not None:
        model.addConstr(gp.quicksum(dev[i] for i in range(m)) <= B)

    obj = gp.quicksum(a_np[i] * a[i] for i in range(m))
    obj += 0.5 * gp.quicksum(G_np[i, j] * a[i] * a[j]
                             for i in range(m) for j in range(m))
    obj += c_edit * gp.quicksum(dev[i] for i in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"gurobi returned status {model.Status} (2 == OPTIMAL); the 'exact' tier "
            "must not fall back to an unproven incumbent -- fix the model or use "
            "solver='brute'.")
    return torch.tensor([float(round(a[i].X)) for i in range(m)])


def _select_exact(alpha, Gamma, c_edit, B, balance, m, L, solver):
    try:
        if solver == "gurobi":
            import gurobipy  # noqa: F401  (probe availability)
            return _select_gurobi(alpha, Gamma, c_edit, B, balance, m, L)
    except ImportError:
        warnings.warn("gurobi unavailable; falling back to brute force.")
    if solver == "brute" or (L + 1) ** m <= 3 ** 12:
        return _select_bruteforce(alpha, Gamma, c_edit, B, balance, m, L)
    # No silent degradation: past that, enumeration is infeasible.
    raise RuntimeError(
        f"no exact solver available for m={m}, L={L} (brute force is (L+1)^m); "
        "install gurobipy or use select_mode='greedy'.")


# --------------------------------------------------------------------------- #
# unified interface
# --------------------------------------------------------------------------- #
def select_action(alpha, Gamma, mode, B=None, c_edit=0.0, balance=True, L=2,
                  solver="gurobi"):
    """mode in {"greedy","exact"}; returns a in {0,...,L}^m, feasible."""
    m = alpha.shape[0]
    if mode == "greedy":
        return _select_greedy(alpha, c_edit, B, balance, m, L)
    if mode == "exact":
        return _select_exact(alpha, Gamma, c_edit, B, balance, m, L, solver)
    raise ValueError(f"unknown mode {mode!r}")
