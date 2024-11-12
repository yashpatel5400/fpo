import argparse
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import os
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


def solve_rayleigh():
    # Parameters
    Lx, Lz = 4, 1
    Nx, Nz = 256, 64
    Rayleigh = 2e6
    Prandtl = 1
    dealias = 3/2
    stop_sim_time = 1
    timestepper = d3.RK222
    max_timestep = 0.125
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background
    bic = b['g'].copy()

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')

    # Main loop
    startup_iter = 10
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    bfc = b.allgather_data('c')

    return (bic.flatten(), bfc.flatten())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    args = parser.parse_args()

    pde_to_func = {
        "poisson":  solve_poisson,
        "rayleigh": solve_rayleigh,
    }

    fs, us = [], []
    N = 1
    for _ in range(N):
        f, u = pde_to_func[args.pde]()
        fs.append(f) 
        us.append(u)
    fs = np.array(fs)
    us = np.array(us)

    os.makedirs(utils.PDE_DIR(args.pde))
    with open(utils.DATA_FN(args.pde), "wb") as f:
        pickle.dump((fs, us), f)