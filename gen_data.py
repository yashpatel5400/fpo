import argparse
import copy
import einops
import math
import logging
import os
import pickle

import scipy.stats.qmc as qmc
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import torch
import math
import scipy.io
from timeit import default_timer
from tqdm.notebook import tqdm

import utils

logger = logging.getLogger(__name__)
device = "cuda:0"

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


def solve_poisson_trial():
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
    
    # HACK: for some reason, the first "sample" is just zeros, so generate 2 and use the second
    scaGen2D = ScalarGenerator2D(size=(Lx, Ly), nCell=(Nx, Ny), nKnot=(8,8))
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


def solve_poisson(N):
    fs, us = [], []
    for _ in range(N):
        f, u = solve_poisson_trial()
        fs.append(f) 
        us.append(u)
    return np.array(fs), np.array(us)


# ------ Somewhat redundant, but slightly different GP processor for generating NS data ----- #
class GaussianRF(object):
    """
    Represents a Gaussian Random Field generator.

    Args:
        dim (int): The dimension of the random field.
        size (int): The size of the random field.
        alpha (float, optional): The parameter alpha. Defaults to 2.
        tau (float, optional): The parameter tau. Defaults to 3.
        sigma (float, optional): The standard deviation of the random field. If None, it is calculated based on tau and alpha. Defaults to None.
        boundary (str, optional): The boundary condition of the random field. Defaults to "periodic".
        device (str, optional): The device to use for computation. Defaults to None.

    Attributes:
        dim (int): The dimension of the random field.
        device (str): The device used for computation.
        sqrt_eig (torch.Tensor): The square root of the eigenvalues of the random field.
        size (tuple): The size of the random field.

    Methods:
        sample(N): Generates N samples of the random field.

    """

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        """
        Initializes a GaussianRF object.

        Args:
            dim (int): The dimension of the random field.
            size (int): The size of the random field.
            alpha (float, optional): The parameter alpha. Defaults to 2.
            tau (float, optional): The parameter tau. Defaults to 3.
            sigma (float, optional): The standard deviation of the random field. If None, it is calculated based on tau and alpha. Defaults to None.
            boundary (str, optional): The boundary condition of the random field. Defaults to "periodic".
            device (str, optional): The device to use for computation. Defaults to None.
        """

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        """
        Generates N samples of the random field.

        Args:
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated samples of the random field.
        """

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real
    

# Function to solve Navier-Stokes equation in 2D
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """
    Solve the 2D Navier-Stokes equations using the Fourier spectral method.

    Parameters:
    - w0 (torch.Tensor): Initial vorticity field.
    - f (torch.Tensor): Forcing field.
    - visc (float): Viscosity coefficient.
    - T (float): Total time.
    - delta_t (float): Time step size (default: 1e-4).
    - record_steps (int): Number of steps between each recorded solution (default: 1).

    Returns:
    - sol (torch.Tensor): Solution tensor containing the vorticity field at each recorded time step.
    - sol_t (torch.Tensor): Time tensor containing the recorded time steps.
    """
    # Grid size - it must be power of 2
    N = w0.size()[-1]

    # Max wavenumber
    k_max = math.floor(N/2.0)

    # Total number of steps
    steps = math.ceil(T/delta_t)

    # Initial vortex field in Fourier space
    w_h = torch.fft.rfft2(w0)

    # Forcing field in Fourier space
    f_h = torch.fft.rfft2(f)

    # If the same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Save the solution every certain number of steps
    record_time = math.floor(steps/record_steps)

    # Wave numbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    
    # Wave numbers in x-direction
    k_x = k_y.transpose(0,1)

    # Remove redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Negative of the Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Save the solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    
    #Record counter
    c = 0
    
    #Physical time
    t = 0.0
    for j in range(steps):
        
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))
            sol[...,c] = w
            c += 1

    return sol


def solve_navier_stokes(N):
    #Resolution
    s = 256

    #Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Time grid
    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    X,Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    #Number of snapshots from solution
    record_steps = 200

    #Inputs
    a = torch.zeros(N, s, s)
    #Solutions
    u = torch.zeros(N, s, s, record_steps)

    #Solve equations in batches (order of magnitude speed-up)
    bsize = 50

    c = 0
    t0 = default_timer()
    for j in range(N // bsize):
        print(f"Batch: {j}")

        #Sample random feilds
        w0 = GRF.sample(bsize)

        #Solve NS
        sol = navier_stokes_2d(w0, f, 1e-5, 2.5, 1e-4, record_steps)

        a[c:(c+bsize),...] = w0
        u[c:(c+bsize),...] = sol

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)
    X, Y = u[...,0], u[...,-1]
    
    # used convert to Dedalus coefficient representation for final dataset
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dtype = np.float64
    coords = d3.CartesianCoordinates("x", "y")
    dist   = d3.Distributor(coords, dtype=dtype)
        
    xbasis = d3.RealFourier(coords["x"], size=s, bounds=(0, Lx))
    ybasis = d3.RealFourier(coords["y"], size=s, bounds=(0, Ly))
    field = dist.Field(name='u', bases=(xbasis, ybasis))

    u_i_cs, u_f_cs = [], []    
    for u_i, u_f in zip(X, Y):
        field["g"] = u_i.detach().cpu().numpy()
        u_i_cs.append(field["c"].copy())

        field["g"] = u_f.detach().cpu().numpy()
        u_f_cs.append(field["c"].copy())
    return np.array(u_i_cs), np.array(u_f_cs)


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
    parser.add_argument("--N", type=int)
    args = parser.parse_args()

    pde_to_func = {
        "poisson":  solve_poisson,
        "navier_stokes": solve_navier_stokes,
        "qgniw": solve_qgniw,
    }
    fs, us = pde_to_func[args.pde](args.N)

    os.makedirs(utils.PDE_DIR(args.pde), exist_ok=True)
    with open(utils.DATA_FN(args.pde), "wb") as f:
        pickle.dump((fs, us), f)