import cvxpy as cp
import numpy as np
import pickle

from calibrate import get_partials
from wavelet import WaveletBasis

def optimize(uhat, quantile):
    wavelet_basis = WaveletBasis()

    # amortized definition of basis partials for Sobolev norm computation
    uhat_coeffs = wavelet_basis.get_decomp(uhat)
    shaped_basis = wavelet_basis.basis_func.reshape(wavelet_basis.basis_func.shape[1], uhat.shape[0], uhat.shape[1])
    partials = get_partials(shaped_basis)
    partials = [
        np.transpose(np.array([partial.reshape(partial.shape[0], -1) for partial in partial_order]), (0, 2, 1)) 
        for partial_order in partials
    ]
    x_grid = wavelet_basis.x_grid.reshape(uhat.shape[0], uhat.shape[1], -1)

    # problem specification (constant over optimization)
    eta       = 1e-4
    r         = 0.1
    max_iters = 100

    w = np.array([0.6, 0.6])
    for iter in range(max_iters):
        # ---- Compute u^* using parametric wavelet formulation
        u_coeff = cp.Variable(uhat_coeffs.shape)

        integral_mask = (np.linalg.norm(wavelet_basis.x_grid - w, axis=1) < r).astype(np.int8).reshape(uhat.shape)
        psi_w = np.sum(shaped_basis * integral_mask, axis=(1,2))
        objective = cp.Minimize(-u_coeff @ psi_w)

        constraints = [
            cp.sum([
                cp.max(cp.hstack([
                    cp.abs(partial @ (uhat_coeffs - u_coeff)) for partial in partial_order
                ])) for partial_order in partials
            ]) <= quantile
        ]

        prob = cp.Problem(objective, constraints)
        obj  = prob.solve()
        u_star = (wavelet_basis.basis_func @ u_coeff.value).reshape(32, 32)

        # ---- Update to w^(t+1) using u^*
        eps = 5e-2
        bd_mask = (np.abs(np.linalg.norm(x_grid - w, axis=-1) - r) < eps).astype(np.int8)
        w_grad_field = np.expand_dims(bd_mask, axis=-1) * (w - x_grid) * np.expand_dims(u_star, axis=-1)
        w_grad = np.sum(w_grad_field, axis=(0, 1))
        w = w - eta * w_grad

        print(f"{iter} : {obj} -- w : {w} -- w_grad : {np.linalg.norm(w_grad)}")
    return w


if __name__ == "__main__":
    fn = "/Users/yppatel/Documents/PhD_Year3_Summer/Research/FPO/data.pkl"
    with open(fn, "rb") as f:
        (yyhat, yybatch) = pickle.load(f)
    subsample = 4
    _, uhat = yybatch[0,::subsample,::subsample,0,0], yyhat[0,::subsample,::subsample,0,0]

    conformal_quantile = 50 # replace with quantile from dataset
    optimize(uhat, conformal_quantile)