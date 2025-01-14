import argparse
import copy
import math
import logging
import os
import pickle

import scipy.stats.qmc as qmc
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import utils

logger = logging.getLogger(__name__)

class ScalarGenerator2D:
    def __init__(self, size=(1.0, 1.0), nCell=(32, 32), nKnot=(4, 4)):
        assert len(size) == 2 and len(nCell) == 2 and len(nKnot) == 2
        assert size[0]/nCell[0] == size[1]/nCell[1]
        # input arguments
        self.nCell  = nCell
        self.size   = size
        self.nKnot  = nKnot

        # module constants
        self.lenScale = 1.0

        # knots info for GP
        xKnot = np.linspace(0.0, self.size[0], self.nKnot[0])
        yKnot = np.linspace(0.0, self.size[1], self.nKnot[1])
        xKnot, yKnot = np.meshgrid(xKnot, yKnot)
        # list of knot's (x, y)
        xyKnot = np.stack([xKnot.flatten(), yKnot.flatten()], axis=-1)
        # squared distance of each knot pair
        knotDistMat = distance.cdist(xyKnot, xyKnot, 'sqeuclidean')
        # knot's colvariance matrix
        knotCovMat    = np.exp(-knotDistMat / self.lenScale)
        self.knotCovMatInv = np.linalg.inv(knotCovMat)

        # setup the output scalar's coordinates and matrix for GP
        h      = self.size[0] / self.nCell[0]
        x      = [(i+0.5)*h for i in range(self.nCell[0])]
        y      = [(i+0.5)*h for i in range(self.nCell[1])]
        x, y   = np.meshgrid(x, y)
        # list of coordinates
        xy     = np.stack([x.flatten(), y.flatten()], axis=-1)
        # colvariance matrix for grid cells and knots
        self.covMat = distance.cdist(xy, xyKnot, 'sqeuclidean')
        self.covMat = np.exp(-self.covMat / self.lenScale)

        # values at the boundary cells' face centers
        xyBc   = np.zeros((2*np.sum(self.nCell), 2))
        nx, ny = self.nCell[0], self.nCell[1]
        # i- boundary
        xyBc[:nx, 0] = np.array([(j+0.5)*h for j in range(nx)])
        # j+ boundary
        xyBc[nx:nx+ny, 0] = size[0]
        xyBc[nx:nx+ny, 1] = np.array([(i+0.5)*h for i in range(ny)])
        # i+ boundary
        xyBc[nx+ny:2*nx+ny, 0] = np.flip(np.array([(j+0.5)*h for j in range(nx)]))
        xyBc[nx+ny:2*nx+ny, 1] = self.size[1]
        # j- boundary
        xyBc[2*nx+ny:, 1] = np.flip(np.array([(i+0.5)*h for i in range(ny)]))
        # colvariance matrix for boundary face centers and knots
        self.covMatBc = distance.cdist(xyBc, xyKnot, 'sqeuclidean')
        self.covMatBc = np.exp(-self.covMatBc / self.lenScale)

    def generate_scalar2d(self, nSample, valMin=0.0, valMax=1.0, outputBc=False,
                            strictMin=False, periodic=0):
        # create sobol sequence
        pow      = int(np.log2(self.nKnot[0]*self.nKnot[1]*nSample)) + 1
        sobolSeq = qmc.Sobol(d=1).random_base2(m=pow)
        sobolSeq = sobolSeq * (valMax - valMin) + valMin
        np.random.shuffle(sobolSeq)

        # allocate the scalars
        samples = np.zeros((nSample, self.nCell[1], self.nCell[0]))
        if outputBc:
          bcs = np.zeros((nSample, 2*np.sum(self.nCell)))

        # generate the scalar with GP
        # R.B.Gramacy P148, Eqn 5.2
        s, e = 0, 0
        for i in range(nSample):
          if periodic:
            period = 30
            A = 2*math.pi/period
            if i%period == 0:
              s, e = e, e + self.nKnot[0] * self.nKnot[1]
            knots = copy.deepcopy(sobolSeq[s:e])
            knots *= (math.sin(i/A-math.pi/2)+1) / 2
          else:
            s, e  = e, e + self.nKnot[0] * self.nKnot[1]
            knots = sobolSeq[s:e]
          # interpolate one scalar with GP
          sca            = np.matmul(self.covMat, np.dot(self.knotCovMatInv, knots))
          samples[i,...] = np.reshape(sca, self.nCell)
          if outputBc:
            bc           = np.matmul(self.covMatBc, np.dot(self.knotCovMatInv, knots))
            bcs[i,:]     = np.squeeze(bc)

        if strictMin:
          scalarMin = min(np.min(bcs), np.min(samples))
          samples   = samples - scalarMin + valMin
          bcs       = bcs     - scalarMin + valMin

        if outputBc:
          return samples, bcs

        return samples

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
    
    nx, ny  = 256, 256
    lx, ly  = 5.0, 5.0
    scaGen2D = ScalarGenerator2D((lx, ly), (nx, ny))

    # HACK: for some reason, the first "sample" is just zeros, so generate 2 and use the second
    f_val    = scaGen2D.generate_scalar2d(2, valMin=-10.0, valMax=10.0, periodic=True)
    f['g'] = f_val[-1]

    # Problem
    problem = d3.LBVP([u, tau_u], namespace=locals())
    problem.add_equation("lap(u) + tau_u = f")
    problem.add_equation("integ(u) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve()

    uc = u.allgather_data('c')
    
    return (f['c'].flatten(), uc.flatten())


def solve_flow():
    # Parameters
    Lx, Lz = 1, 2
    Nx, Nz = 128, 256
    Reynolds = 5e4
    Schmidt = 1
    dealias = 3/2
    stop_sim_time = 3
    timestepper = d3.RK222
    max_timestep = 1e-2
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist   = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    s = dist.Field(name='s', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')

    # Substitutions
    nu = 1 / Reynolds
    D = nu / Schmidt
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)

    # Problem
    problem = d3.IVP([u, s, p, tau_p], namespace=locals())
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
    problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time
    
    # Initial conditions
    # Background shear
    u['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
    # Match tracer to shear
    s['g'] = u['g'][0]
    # Add small vertical velocity perturbations localized to the shear layers
    u['g'][1] += np.random.random() * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
    snapshots.add_task(s, name='tracer')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property((u@ez)**2, name='w2')

    # Main loop
    uic = np.linalg.norm(u['c'].copy(), axis=0)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_w = np.sqrt(flow.max('w2'))
                logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
    ufc = np.linalg.norm(u['c'].copy(), axis=0)    

    return (uic.flatten(), ufc.flatten())


def solve_qgniw():
    # Numerics Parameters
    L = 10; Lx, Ly = L, L
    log_n = 9; Nx, Ny = 2**log_n, 2**log_n
    dtype = np.float64

    dealias = 3/2
    stop_sim_time = 25/(2*np.pi)
    timestepper = d3.RK443
    dtype = np.float64

    #Physical Parameters
    kap = 5e-8*((2**(10-log_n))**4)

    # Bases
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

    # Fields
    zeta = dist.Field(bases=(xbasis,ybasis) )
    psi = dist.Field(bases=(xbasis,ybasis) )
    tau_psi = dist.Field()

    # Substitutions
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])
    lap = lambda A: d3.Laplacian(A)
    integ = lambda A: d3.Integrate(A, ('x', 'y'))

    x, y = dist.local_grids(xbasis, ybasis)

    J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)
    l4H = lambda A: lap(lap(A))

    KE = integ(dx(psi)**2+dy(psi)**2)/2
    Enstrophy = integ(zeta**2)/2

    # Problem
    problem = d3.IVP([zeta, psi, tau_psi], namespace=locals())
    problem.add_equation("lap(psi) + tau_psi = zeta")
    problem.add_equation("dt(zeta) + kap*l4H(zeta) = - J(psi,zeta)")
    problem.add_equation("integ(psi) = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    zeta.fill_random('c', distribution='normal', scale=6.2e-2) # Random noise
    # Filter the IC
    kx = xbasis.wavenumbers[dist.local_modes(xbasis)]; ky = ybasis.wavenumbers[dist.local_modes(ybasis)]; K = np.sqrt(kx**2+ky**2)
    init_fac = K*(1+(K/(2*np.pi))**4)**(-1/2)
    zeta['c'] *= init_fac

    # Analysis
    snapdata = solver.evaluator.add_file_handler('2DEuler_snap', sim_dt=0.1, max_writes=50)
    snapdata.add_task(-(-zeta), name='ZETA')
    snapdata.add_task(-(-psi), name='PSI')

    diagdata = solver.evaluator.add_file_handler('2DEuler_diag', sim_dt=0.01, max_writes=stop_sim_time*100)
    diagdata.add_task(KE, name='KE')
    diagdata.add_task(Enstrophy, name='Enstrophy')

    # Flow properties
    dt_change_freq = 10
    flow = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
    flow.add_property(abs(dy(psi)), name='absu')
    flow.add_property(abs(dx(psi)), name='absv')
    flow.add_property(-psi*zeta/2, name='KE')

    # Main loop
    timestep = 1e-7
    delx = Lx/Nx; dely = Ly/Ny

    uic = zeta['c'].copy()
    print(uic.shape)
    try:
        logger.info('Starting main loop')
        solver.step(timestep)
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % dt_change_freq == 0:
                maxU = max(1e-10,flow.max('absu')); maxV = max(1e-10,flow.max('absv'))
                timestep_CFL = min(delx/maxU,dely/maxV)*0.5
                timestep = min(max(1e-5, timestep_CFL), 1)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE')))

    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
    ufc = zeta['c'].copy()

    return (uic.flatten(), ufc.flatten())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    args = parser.parse_args()

    pde_to_func = {
        "poisson":  solve_poisson,
        "rayleigh": solve_flow,
        "qgniw": solve_qgniw,
    }

    fs, us = [], []
    N = 300
    for _ in range(N):
        f, u = pde_to_func[args.pde]()
        fs.append(f) 
        us.append(u)
    fs = np.array(fs)
    us = np.array(us)

    os.makedirs(utils.PDE_DIR(args.pde), exist_ok=True)
    with open(utils.DATA_FN(args.pde), "wb") as f:
        pickle.dump((fs, us), f)