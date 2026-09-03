"""Per-point scores alpha and interaction matrix B  (Brief II, sec. 3-4.2, Fact 1).

    Q_t(theta, z, a) = C_t + c_edit sum_i|a_i-1| + sum_i a_i alpha_i
                       + 1/2 sum_ij a_i a_j B_ij + R(a),   |R| = O(eta^3)

    alpha_i = lam <e, phi_i>            (immediate)
            + <grad V_bar(theta), phi_i>   (cost-to-go)     , e = theta - theta_target

    B_ij    = lam <phi_i, phi_j>        (exact Gram part)
            + phi_i^T H V_bar(theta) phi_j (2nd order)

The Gram part is exact and cheap; only the Hessian part needs autodiff (via
Hessian-vector products).  Set use_hessian=False for the Gram-only ablation.
"""
import torch


def scores_and_interaction(V_psi, theta, z, t, victim, theta_target, lam,
                           need_B=True, use_hessian=True):
    """Returns (alpha (m,), B (m,m) or None), both detached."""
    Phi = victim.phi_batch(theta, z)               # (m, d): rows phi_i, treated as constants
    e = theta - theta_target                       # (d,)

    # --- gradient (and, on demand, Hessian) of the *integrated* value V_bar at t+1 ---
    # NOTE the t+1: alpha uses grad V_hat(., t+1), not t. Off-by-one here is the classic bug.
    theta_ = theta.detach().requires_grad_(True)
    v = V_psi(theta_, t + 1)
    gradV, = torch.autograd.grad(v, theta_, create_graph=use_hessian)  # (d,)

    alpha = lam * (Phi @ e) + (Phi @ gradV)        # (m,)
    if not need_B:
        return alpha.detach(), None

    B = lam * (Phi @ Phi.T)                         # exact Gram part (closed form)
    if use_hessian:
        # HVP: d/dtheta <gradV, phi_j> == H @ phi_j  (double backward).
        HPhi = torch.stack([
            torch.autograd.grad(gradV @ Phi[j], theta_, retain_graph=True)[0]
            for j in range(Phi.shape[0])
        ])                                          # (m, d): row j = H @ phi_j
        B = B + Phi @ HPhi.T
    return alpha.detach(), B.detach()
