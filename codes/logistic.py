from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def logistic(t: float, y: ArrayLike, args: tuple[float, float]) -> ArrayLike:
    """Logistic growth model.

    Args:
        t: time
        y: dependent variable
        args: parameters
        
    Returns:
        dydt: derivative of y
    """
    r, K = args
    return r * y * (1 - y / K)

if __name__ == '__main__':
    # parameters
    r = 0.3
    K = 100.0
    args = (r, K)
    y0 = jnp.array(0.5 * K) # initial value
    t0, t1, dt = 0, 10, 0.1 # init time, final time, time step
    ts = jnp.arange(t0, t1, dt) # time steps

    # use diffrax to solve diff eq.
    term = ODETerm(logistic)
    solver = Dopri5()
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
    print(sol.ys)
