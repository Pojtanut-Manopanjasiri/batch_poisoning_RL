import itertools
import warnings

import numpy as np
import torch


def _objective(a, alpha, Gamma, rho):
    edits = torch.abs(a - 1.0).sum()
    quad = 0.0 if Gamma is None else 0.5 * a @ Gamma @ a
    return float(a @ alpha + quad + rho * edits)


def _feasible(a, balance, B, L):
    if bool((a < 0).any()) or bool((a > L).any()):
        return False
    if balance and int(a.sum()) != len(a):
        return False
    if B is not None and float((a - 1.0).abs().sum()) > B:
        return False
    return True


def _select_greedy(alpha, rho, B, balance, m, L):
    if not balance:
        a = torch.ones(m)
        a[alpha > rho] = 0.0
        a[alpha < -rho] = float(L)
        if B is not None:
            cost = (a - 1.0).abs()                      
            gain = alpha * 1.0 - alpha * a              
            gain = gain + rho * (1.0 - cost)
            spent, order = 0.0, torch.argsort(gain, descending=True)
            keep = torch.ones(m)
            for i in order:
                if spent + float(cost[i]) <= B:
                    keep[i] = a[i]
                    spent += float(cost[i])
            a = keep
        return a

    k = m if B is None else B // 2       
    a = torch.zeros(m)
    D = 0                              
    for _ in range(m):                  
        d = alpha + torch.where(a == 0, -rho, rho)
        elig = (a < L) & ((a == 0) | (D < k))
        d = d.masked_fill(~elig, float("inf"))
        i = int(torch.argmin(d))
        if a[i] >= 1:
            D += 1
        a[i] += 1
    return a


def select_greedy_batch(alpha, lam=0.0, B=None, balance=True, L=2):
    S, m = alpha.shape
    rho = lam / (2 * m)                
    if not balance:                     
        return torch.stack([_select_greedy(alpha[s].cpu(), rho, B, balance, m, L)
                            for s in range(S)]).to(alpha.device)
    k = m if B is None else B // 2
    a = torch.zeros_like(alpha)
    D = torch.zeros(S, device=alpha.device)
    rows = torch.arange(S, device=alpha.device)
    for _ in range(m):                   
        d = alpha + torch.where(a == 0, -rho, rho)
        elig = (a < L) & ((a == 0) | (D < k)[:, None])
        d = d.masked_fill(~elig, float("inf"))
        i = torch.argmin(d, dim=1)
        D += (a[rows, i] >= 1).float()
        a[rows, i] += 1
    return a

_FEASIBLE_CACHE = {}


def _feasible_set(m, balance, B, L):
    key = (m, balance, B, L)
    if key not in _FEASIBLE_CACHE:
        acts = [torch.tensor(c, dtype=torch.float32)
                for c in itertools.product(*[range(L + 1)] * m)
                if _feasible(torch.tensor(c, dtype=torch.float32), balance, B, L)]
        _FEASIBLE_CACHE[key] = torch.stack(acts)
    return _FEASIBLE_CACHE[key]


def count_feasible(m, balance, B, L):
    states = {(0, 0): 1}
    for _ in range(m):
        nxt = {}
        for (s, e), c in states.items():
            for v in range(L + 1):
                e2 = e + abs(v - 1)
                if B is None or e2 <= B:
                    nxt[(s + v, e2)] = nxt.get((s + v, e2), 0) + c
        states = nxt
    return sum(c for (s, _), c in states.items() if not balance or s == m)


def _select_bruteforce(alpha, Gamma, rho, B, balance, m, L):
    A = _feasible_set(m, balance, B, L)
    edits = torch.abs(A - 1.0).sum(1)
    quad = 0.5 * (A @ Gamma * A).sum(1)
    obj = A @ alpha + quad + rho * edits
    return A[int(torch.argmin(obj))].clone()


def _select_gurobi(alpha, Gamma, rho, B, balance, m, L):
    import gurobipy as gp
    from gurobipy import GRB

    a_np, G_np = alpha.numpy(), Gamma.numpy()
    model = gp.Model("iqp")
    model.Params.OutputFlag = 0
  
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.IntFeasTol = 1e-9
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
    G_sym = 0.5 * (G_np + G_np.T).astype(float)
    w, U = np.linalg.eigh(G_sym)
    tol = 1e-5 * max(abs(w).max(), 1e-30)          
    if w.min() < -tol:
        raise ValueError(f"Gamma must be PSD (the Gram Phi Phi^T); min eigenvalue {w.min():.3g}")
    
    ks = np.where(w > tol)[0]                     
    u = model.addVars(len(ks), lb=-GRB.INFINITY)
    for j, k in enumerate(ks):
        model.addConstr(u[j] == gp.quicksum(float(U[i, k]) * a[i] for i in range(m)))
    obj += 0.5 * gp.quicksum(float(w[k]) * u[j] * u[j] for j, k in enumerate(ks))
    obj += rho * gp.quicksum(dev[i] for i in range(m))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"gurobi returned status {model.Status} (2 == OPTIMAL); the 'exact' tier "
            "must not fall back to an unproven incumbent -- fix the model or use "
            "solver='brute'.")
    return torch.tensor([float(round(a[i].X)) for i in range(m)])


def _select_exact(alpha, Gamma, rho, B, balance, m, L, solver):
    try:
        if solver == "gurobi":
            import gurobipy  # noqa: F401  (probe availability)
            return _select_gurobi(alpha, Gamma, rho, B, balance, m, L)
    except ImportError:
        warnings.warn("gurobi unavailable; falling back to brute force.")
    if solver == "brute" or (L + 1) ** m <= 3 ** 12:
        return _select_bruteforce(alpha, Gamma, rho, B, balance, m, L)
    # No silent degradation: past that, enumeration is infeasible.
    raise RuntimeError(
        f"no exact solver available for m={m}, L={L} (brute force is (L+1)^m); "
        "install gurobipy or use select_mode='greedy'.")


def select_action(alpha, Gamma, mode, B=None, lam=0.0, balance=True, L=2,
                  solver="gurobi"):
    """mode in {"greedy","exact"}; returns a in {0,...,L}^m, feasible."""
    device = alpha.device
    alpha = alpha.cpu()
    Gamma = None if Gamma is None else Gamma.cpu()
    m = alpha.shape[0]
    rho = lam / (2 * m)                  # Algorithm 2, line 1
    if mode == "greedy":
        a = _select_greedy(alpha, rho, B, balance, m, L)
    elif mode == "exact":
        a = _select_exact(alpha, Gamma, rho, B, balance, m, L, solver)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return a.to(device)
