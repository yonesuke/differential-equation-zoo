from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def damped_oscillator(t: float, y: ArrayLike, args: tuple[float, float, float]) -> ArrayLike:
    """Damped oscillator model.

    Args:
        t: time
        y: dependent variable
        args: parameters
        
    Returns:
        dydt: derivative of y
    """
    mass, k, alpha = args
    return jnp.array([y[1], -k / mass * y[0] - alpha / mass * y[1]])

if __name__ == '__main__':
    # parameters
    mass = 1.0
    k = 1.0
    alpha = 0.1
    args = (mass, k, alpha)
    y0 = jnp.array([1.0, 0.0]) # initial value
    t0, t1, dt = 0, 10, 0.1 # init time, final time, time step
    ts = jnp.arange(t0, t1, dt) # time steps
    
    # use diffrax to solve diff eq.
    term = ODETerm(damped_oscillator)
    solver = Dopri5()
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
    print(sol.ys)
