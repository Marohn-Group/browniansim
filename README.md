# browniansim

1D Brownian-motion simulation toolkit for a damped harmonic oscillator (cantilever-like system), with numerical and analytical solvers plus demo notebooks for frequency-noise workflows.

## Features

- Simulate damped harmonic oscillator motion with thermal force and optional shot noise.
- Demo notebooks for noise generation, pulse response, frequency shift, parameter amplification, and fitting.
- Unit tests validating numerical behavior against RK4 and analytical references.

## Repository Layout

```
browniansim/
	__init__.py
	simulator.py
demo/
	demo1_noiseGenerator.ipynb
	demo2_CantileverResponse2HardPulse.ipynb
	demo3_CantileverResponse2OscilattngPulse.ipynb
	demo4_FrequecnyShift.ipynb
	demo5_ParameticAmplification.ipynb
	demo6_ChangingT4Brownian.ipynb
	demo7_FittingParametersWithScipy.ipynb
	demo8_concatingSignals.ipynb
	function/
		fitter.py
		RussellExtractF.py
tests/
	test_simulator.py
requirements.txt
requirements-demo.txt
```

## Installation

### Core package dependencies

```bash
pip install -r requirements.txt
```

### Demo dependencies

```bash
pip install -r requirements-demo.txt
```

## Quick Start

```python
import numpy as np
import browniansim as bs

# Example physical / simulation parameters
T = 275.0                 # K
k = 2.7e-3                # N/m
Q = 18000.0
dtau = 0.05 * 2 * np.pi   # normalized timestep (dt * w0)
X0 = [0.0, 0.0]           # [x0, v0/w0]
N = 4000

# Frequency shift array for RK4 sampler stages, shape (N, 4)
dw = np.zeros((N, 4))

# Optional additive measurement noise standard deviation
std_shotN = 1e-10

traj = bs.brownian_simulator(
		T=T,
		k=k,
		Q=Q,
		dtau=dtau,
		X0=X0,
		dw=dw,
		std_shotN=std_shotN,
		withShotNoise=True,
)

x = traj[:, 0]
v = traj[:, 1]
```

## Main API

- `brownian_simulator(T, k, Q, dtau, X0, dw, std_shotN, withShotNoise=True)`
	Simulates noisy oscillator trajectory and returns an `(N, 2)` array: displacement and normalized velocity.

- `solve_RK4(X0, dtau, Q, F, dw)`
	RK4 integrator for the damped oscillator, where `F` and `dw` have shape `(N, 4)`.

- `harmosc_shiftw(X, Q, F, dw)`
	Differential equation right-hand side for shifted-frequency damped oscillator dynamics.

- `analytical_sol_ringdown(X0, Q, dw, dtau, p0=0)`
	Analytical damped solution, returns stacked time and displacement.

- `analytical_sol_cons(X0, Q, dw, dtau, p0=0)`
	Analytical conservative (non-damped) solution.

- `pos2sig(pos, lam, V0, Vb, phi0=-np.pi/2)`
	Converts displacement trajectory to interferometric voltage-like signal.

## Demos

Notebook workflows are under `demo/` and include:

- Noise generation
- Response to hard and oscillating pulses
- Frequency-shift analysis
- Parametric amplification
- Temperature dependence
- Parameter fitting with SciPy
- Signal concatenation

Helper utilities for fitting and demodulation are in `demo/function/`.

## Testing

Run the test suite with:

```bash
pytest -q
```

Current tests in `tests/test_simulator.py` check:

- Brownian simulator consistency with RK4 when thermal and shot noise are disabled.
- Agreement with analytical ringdown behavior.
- Expected heating-up signal statistics.

## License

MIT License. See `LICENSE`.
