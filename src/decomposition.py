import torch

def alpha_batch(V_psi, theta, t_next, Phi, theta_target):
    theta_ = theta.detach().requires_grad_(True)
    gradV, = torch.autograd.grad(V_psi(theta_, t_next).sum(), theta_)      # (S, d)
    e = theta - theta_target
    return torch.einsum("smd,sd->sm", Phi, e + gradV).detach()


def scores_and_interaction(V_psi, theta, z, t, victim, theta_target, need_Gamma=True):
    Phi = victim.phi_batch(theta, z)               
    e = theta - theta_target                      

    theta_ = theta.detach().requires_grad_(True)
    gradV, = torch.autograd.grad(V_psi(theta_, t + 1), theta_)   

    alpha = Phi @ (e + gradV)                     
    if not need_Gamma:
        return alpha.detach(), None
    return alpha.detach(), (Phi @ Phi.T).detach()              
