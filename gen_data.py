import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import pickle

import utils

logger = logging.getLogger(__name__)

def solve_poisson():
    # Parameters
    Lx, Ly = 2 * np.pi, 2 * np.pi
    Nx, Ny = 256, 256
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates("x", "y")
    dist   = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx))
    ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly))

    # Fields
    u = dist.Field(name='u', bases=(xbasis, ybasis))
    tau_u = dist.Field(name='tau_u')

    # Forcing
    f = dist.Field(name='f', bases=(xbasis, ybasis))
    f.fill_random('g')

    # Problem
    problem = d3.LBVP([u, tau_u], namespace=locals())
    problem.add_equation("lap(u) + tau_u = f")
    problem.add_equation("integ(u) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve()

    uc = u.allgather_data('c')
    
    return (f['c'].flatten(), uc.flatten())


if __name__ == "__main__":
    pde = "poisson"
    pde_to_func = {
        "poisson": solve_poisson,
    }

    fs, us = [], []
    N = 1_000
    for _ in range(N):
        f, u = pde_to_func[pde]()
        fs.append(f) 
        us.append(u)
    fs = np.array(fs)
    us = np.array(us)

    with open(utils.DATA_FN(pde), "wb") as f:
        pickle.dump((fs, us), f)