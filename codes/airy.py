from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def airy(t: float, y: ArrayLike, args = None) -> ArrayLike:
    """Airy model.

    Args:
        t: time
        y: dependent variable
        args: parameters, not used in this model
        
    Returns:
        dydt: derivative of y
    """
    return jnp.array([y[1], t * y[0]])

if __name__ == '__main__':
    # parameters
    y0 = jnp.array([1.0, 0.0]) # initial value
    t0, t1, dt = 0, 10, 0.1 # init time, final time, time step
    ts = jnp.arange(t0, t1, dt) # time steps
    
    # use diffrax to solve diff eq.
    term = ODETerm(airy)
    solver = Dopri5()
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(term, solver, t0, t1, dt, y0, saveat=saveat)
    print(sol.ys)
