import numpy as np


def harmosc_shiftw(X, Q, F,dw):
    """
    Damped harmonic oscillator with random driving force. Note: this differential equation treats time as time*w0, displacement as x, and momentum as velocity/w0.
    Parameters:
        X[array]: displacement, velocity.
        Q[float]: quality factor.
        F[array]: force devided by mass and w0 squared.
        dw[array]: frequency shift, dw = (w-w0)/w0.
        x[float]: displacement.
        p[float]: velocity/w0.
        dotx[float]: change of displacement.
        dotp[float]: change of velocity.
    Return:
        dotx, dotp[array]: change of displacement and velocity/w0.
    """
    x, p = X
    dotx = p
    dotp = - (1+dw)**2 * x - (1 / Q) * p + F
    return np.array([dotx, dotp])

def solve_RK4(X0, dtau, Q, F,dw):
    """
    Solve the damped harmonic oscillator equation using the Runge-Kutta 4th order method. This function is designed for the case where the frequency has a rapid change, so in our RK4 method, we need to simulate the frequency at t+dt/2.
    Parameters:
        X0[array[2]]: initial displacement, velocity devided by w0.
        dtau[float]: time step*w0.
        Q[float]: quality factor.
        F[array[n,4]]: force devided by the mass and w0 squared.
        dw[array[n,4]]: effective frequency shift, dw = (w-w0)/w0.
    Return:
        X[array]: displacement, velocity devided by w0.
    """
    if len(X0) != 2:
        raise ValueError("X0 must have 2 elements")
    if F.ndim != 2 or F.shape[1] != 4:
        raise ValueError("F must be a 2D array with 4 columns")
    if dw.ndim != 2 or dw.shape[1] != 4:
        raise ValueError("dw must be a 2D array with 4 columns")
    if np.shape(dw)[0] != np.shape(F)[0]:
        raise ValueError("dw and F must have the same length")
    X = np.zeros([np.shape(F)[0], 2])  # F is the force time-series. For the Euler method, each result corresponds to one F. For RK4, we need to use four F values for each step!
    X[0] = X0
    for n in range(np.shape(F)[0]-1):
        k1 = harmosc_shiftw(X[n], Q, F[n][0],dw[n][0])
        k2 = harmosc_shiftw(X[n] + dtau * k1 / 2, Q, F[n][1],dw[n][1])
        k3 = harmosc_shiftw(X[n] + dtau * k2 / 2, Q, F[n][2],dw[n][2])
        k4 = harmosc_shiftw(X[n] + dtau * k3, Q, F[n][3],dw[n][3])
        X[n+1, :] =  X[n, :] + dtau * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X

def brownian_simulator(T, k, Q, dtau, X0, dw, std_shotN, withShotNoise=True):
    r"""
    Parameters:
    T[float]: temperature(K)
    k[float]: spring constant(N/m)
    Q[float]: quality factor
    f0[float]: resonant frequency(Hz)
    dtau[float]: normalized time step: dt/w0. Time is normalized to the period of the resonator.
    X0[list[2]]: initial position [x0/w0,v0], in which x0 is the initial position and v0 is the initial velocity, w0 is the resonant frequency
    dw[np.array[N]]: frequency shift array, the real frequency is w0*(1+dw), so dw =angular velocity shift/w0; corresponding to freq shift in the time domain. Length of dw determine the measurement time.
    std_shotN[float]: standard deviation of the shot noise(m)
    withShotNoise[bool]: whether to add shot noise

    Functions:
    This function is used to simulate the brownian motion of a cantilever.

    Returns:
        noise[np.array]: noise array[N,2], in which the first column is the position and the second column is the velocity
    """

    kb = 1.38 * 10**-23  # boltzman constant
    std_F = np.sqrt(
        4 * kb * T / (Q * k * dtau)
    )  # the standard deviation of the thermal force
    F_row = np.random.normal(0, std_F, size=(len(dw), 1))
    F = np.tile(F_row, (1, 4))
    noise_eq = solve_RK4(X0, dtau, Q, F, dw)
    if withShotNoise:
        noise_shot = np.random.normal(0, std_shotN, size=np.shape(noise_eq))
        noise = noise_eq + noise_shot
    else:
        noise = noise_eq
    return noise


def analytical_sol_ringdown(X0,Q,dw,dtau,p0=0):
    """
    Analytical solution of the damped harmonic oscillator equation.
    Parameters:
    X0[array]: initial amplitude, x*w0.
    Q[float]: quality factor.
    dw[array]: frequency shift vs time, shape (N,), delta w/w0.
    dtau[float]: time step, notice that the period is 2*pi.
    p0[float]: initial phase.
    Returns:
    t[array]: time array, shape (N,).
    x[array]: displacement array, shape (N,).
    """
    phase = np.cumsum(dtau*(1+dw))+p0
    phase = np.concatenate([np.array([p0]),phase[:-1]])
    t = np.arange(0, len(dw) * dtau, dtau)
    x = X0[0] * np.exp(-t / Q / 2) * np.cos(phase * np.sqrt(-1 / Q**2 / 4 + 1)) + X0[1] * np.exp(-t / Q / 2) * np.sin(phase * np.sqrt(-1 / Q**2 / 4 + 1))
    return np.vstack((t,x)).T


def analytical_sol_cons(X0,Q,dw,dtau,p0=0):
    """
    Analytical solution of the non-damped harmonic oscillator equation.
    Parameters:
    X0[array]: initial amplitude, x*w0.
    Q[float]: quality factor.
    dw[array]: frequency shift vs time, shape (N,), delta w/w0.
    dtau[float]: time step, notice that the period is 2*pi.
    p0[float]: initial phase.
    Returns:
    t[array]: time array, shape (N,).
    x[array]: displacement array, shape (N,).
    """
    phase = np.cumsum(dtau*(1+dw))+p0
    phase = np.concatenate([np.array([p0]),phase[:-1]])
    t = np.arange(0, len(dw) * dtau, dtau)
    x = X0[0] * np.cos(phase)+X0[1] * np.sin(phase)
    return np.vstack((t,x)).T


def pos2sig(pos,lam,V0,Vb,phi0=-np.pi/2):
    """
    Convert position to signal. We use interferometry to measure displacement.
    pos[array]: position array, shape (N,).
    lam[float]: wavelength.
    V0[float]: amplitude of the signal.
    Vb[float]: baseline of the signal.
    phi0[float]: initial phase of the signal.
    Returns:
    sig[array]: signal array, shape (N,).
    """
    return V0*np.cos(2*np.pi*pos/lam+phi0)+Vb