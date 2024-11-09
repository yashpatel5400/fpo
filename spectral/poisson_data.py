import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import pickle

logger = logging.getLogger(__name__)

def solve_random_setup():
    # Parameters
    Lx, Ly = 2*np.pi, np.pi
    Nx, Ny = 256, 128
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
    ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(0, Ly))

    # Fields
    u = dist.Field(name='u', bases=(xbasis, ybasis))
    tau_1 = dist.Field(name='tau_1', bases=xbasis)
    tau_2 = dist.Field(name='tau_2', bases=xbasis)

    # Forcing
    x, y = dist.local_grids(xbasis, ybasis)
    f = dist.Field(bases=(xbasis, ybasis))
    g = dist.Field(bases=xbasis)
    f.fill_random('g', seed=40)
    f['c'][10:,10:] = 0
    g['g'] = 0

    # Substitutions
    dy = lambda A: d3.Differentiate(A, coords['y'])
    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    # Problem
    problem = d3.LBVP([u, tau_1, tau_2], namespace=locals())
    problem.add_equation("lap(u) + lift(tau_1,-1) + lift(tau_2,-2) = f")
    problem.add_equation("u(y=0) = g")
    problem.add_equation("u(y=Ly) = g")

    # Solver
    solver = problem.build_solver()
    solver.solve()

    uc = u.allgather_data('c')
    
    return (f['c'].flatten(), uc.flatten())

if __name__ == "__main__":
    fs, us = [], []
    N = 1_000
    for _ in range(N):
        f, u = solve_random_setup()
        fs.append(f) 
        us.append(u)
    fs = np.array(fs)
    us = np.array(us)

    with open("poisson.pkl", "wb") as f:
        pickle.dump((fs, us), f)