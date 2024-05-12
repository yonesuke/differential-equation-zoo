from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def lotka_volterra(t: float, y: ArrayLike, args: tuple[float, float, float, float]) -> ArrayLike:
    """Lotka-Volterra model.

    Args:
        t: time
        y: dependent variable
        args: parameters
        
    Returns:
        dydt: derivative of y
    """
    alpha, beta, gamma, delta = args
    return jnp.array([alpha * y[0] - beta * y[0] * y[1], gamma * y[0] * y[1] - delta * y[1]])

if __name__ == '__main__':
    # parameters
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 1.2
    args = (alpha, beta, gamma, delta)
    y0 = jnp.array([1.0, 5.0]) # initial value
    t0, t1, dt = 0, 10, 0.1 # init time, final time, time step
    ts = jnp.arange(t0, t1, dt) # time steps

    # use diffrax to solve diff eq.
    term = ODETerm(lotka_volterra)
    solver = Dopri5()
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
    # calculate conservation value
    conservation = gamma * sol.ys[:, 0] + beta * sol.ys[:, 1] - alpha * jnp.log(sol.ys[:, 1]) - delta * jnp.log(sol.ys[:, 0])
    print(conservation)
    print(jnp.isclose(conservation, conservation[0] * jnp.ones_like(conservation)).all())
