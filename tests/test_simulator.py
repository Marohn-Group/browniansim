import numpy as np
import browniansim as fc
import pytest


def _browniana_vs_RK4(Q):
    dtau = 0.05*2*np.pi #our sampling rate is about 200kHz, w0 is about 8k, so about 20 times bigger
    dw = np.array([]) # we need to set the efficient rad freq for eacxh set, w_eff = 1+dw, and real rad freq w(t) = w0*w_eff = w0*(1+dw)
    k = 2.7*10**-3
    kb = 1.38*10**-23
    T = 0 # setting temp to be 0k, so there is no thermal jitter
    std_shotN = 0

    n = 50.3#how many cycle in each pulse
    X0_0 = [1,0]# initial displacement and velocity for simulation
    p0 = np.pi/2

    dw_l_ana = np.zeros(int(n*2*np.pi/dtau))#w = 1+dw
    dw_l_sim = np.zeros((int(n*2*np.pi/dtau),4))#w = 1+dw

    noise_browniana = fc.brownian_simulator(T, k, Q, dtau, X0_0, dw_l_sim, std_shotN, withShotNoise=False)
    noise_RK4 = fc.solve_RK4(X0_0, dtau, Q, np.zeros((len(dw_l_sim),4)),dw_l_sim)
    return np.all(np.isclose(noise_browniana, noise_RK4, atol=1e-6))

def test_browniana_is_smae_with_RK4():
    """
    Verify that the browniana simulator is the same as the RK4 simulator.
    """
    for _ in range(3):
        Q = np.random.randint(100, 10000)
        assert _browniana_vs_RK4(Q)



def _browniana_vs_analytical(Q):
    dtau = 0.05*2*np.pi #our sampling rate is about 200kHz, w0 is about 8k, so about 20 times bigger
    dw = np.array([]) # we need to set the efficient rad freq for eacxh set, w_eff = 1+dw, and real rad freq w(t) = w0*w_eff = w0*(1+dw)

    kb = 1.38*10**-23
    k = 2.7*10**-3
    T = 0 # setting temp to be 0k, so there is no thermal jitter
    std_shotN = 0

    n = 50.3#how many cycle in each pulse
    X0_0 = [1,0]# initial displacement and velocity for simulation
    p0 = 0

    dw_l_ana = np.zeros(int(n*2*np.pi/dtau))#w = 1+dw
    dw_l_sim = np.zeros((int(n*2*np.pi/dtau),4))#w = 1+dw

    noise_browniana = fc.brownian_simulator(T, k, Q, dtau, X0_0, dw_l_sim, std_shotN, withShotNoise=False)[:,0]
    noise_analytical = fc.analytical_sol_ringdown(X0_0,Q,dw_l_ana,dtau,p0)[:,1]
    return np.all(np.isclose(noise_browniana, noise_analytical, atol=5e-2))

def test_browniana_is_close_to_analytical():
    """
    Verify that the browniana simulator is very close to the analytical solution.
    """
    for _ in range(3):
        Q = np.random.randint(1000, 10000)
        assert _browniana_vs_analytical(Q)


def test_heatingup_signal():
    """
    Verify that the heating up signal has the correct expected value from the analytical solution.
    """
    Q = 18000
    dtau = 0.05 * 2 * np.pi  # Our sampling rate is about 200kHz, w0 is about 8k, so about 20 times larger
    dw = np.array([])  # We need to set the effective radial frequency for each set, w_eff = 1+dw, and the real radial frequency w(t) = w0*w_eff = w0*(1+dw)

    kb = 1.38 * 10**-23
    T = 275  # Set temperature to 0K, so there is no thermal jitter
    k = 2.7 * 10**-3
    w0 = 2 * np.pi * 8169.7
    std_shotN = 0.1*10**-9

    n = 50.3  # Number of cycles in each pulse
    X0_0 = [1, 0]  # Initial displacement and velocity for simulation
    p0 = np.pi / 2

    dw_l_ana = np.zeros(int(n * 2 * np.pi / dtau))  # w = 1+dw
    dw_l_sim = np.zeros((int(n * 2 * np.pi / dtau),4))  # w = 1+dw
    N = 128
    noise_negon_sqm = 0

    sig_analytical = fc.analytical_sol_ringdown(X0_0, Q, dw_l_ana, dtau, p0)  # While generating the analytical trajectory, we also generate the correct time series
    sig_analytical[:, 0] = sig_analytical[:, 0] / w0
    t = sig_analytical[:, 0]

    for i in range(N):
        X0_0 = [0,0]
        noise_negon = fc.brownian_simulator(T, k, Q, dtau, X0_0, dw_l_sim, std_shotN, withShotNoise=False)
        noise_negon_sqm += noise_negon[:,0]**2/N

    y_negon = kb * T*2 / k  * (1 - np.exp(-t *w0/ Q)) 

    assert np.all(np.isclose(noise_negon_sqm, y_negon, rtol=5e-2))