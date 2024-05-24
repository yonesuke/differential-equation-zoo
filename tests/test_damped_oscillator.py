import unittest
from codes.damped_oscillator import damped_oscillator

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, vmap
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def exact_sol(t: ArrayLike, y0: ArrayLike, args: tuple[float, float, float]) -> ArrayLike:
    mass, k, alpha = args
    omega = jnp.sqrt(k / mass)
    gamma = alpha / (2 * mass)
    if gamma < omega:
        freq = jnp.sqrt(omega**2 - gamma**2)
        const1 = y0[0]
        const2 = (y0[1] + gamma * y0[0]) / freq
        position_fn = lambda t: jnp.exp(-gamma * t) * (const1 * jnp.cos(freq * t) + const2 * jnp.sin(freq * t))
    elif gamma == omega:
        const1 = y0[0]
        const2 = y0[1] + gamma * y0[0]
        position_fn = lambda t: (const1 + const2 * t) * jnp.exp(-gamma * t)
    else:
        amplitude1 = 0.5 * (y0[0] + (gamma * y0[0] + y0[1]) / jnp.sqrt(gamma**2 - omega**2))
        amplitude2 = 0.5 * (y0[0] - (gamma * y0[0] + y0[1]) / jnp.sqrt(gamma**2 - omega**2))
        position_fn = lambda t: jnp.exp(-gamma * t) * (amplitude1 * jnp.exp(jnp.sqrt(gamma**2 - omega**2) * t) + amplitude2 * jnp.exp(-jnp.sqrt(gamma**2 - omega**2) * t))
    velocity_fn = vmap(grad(position_fn))
    return jnp.vstack([position_fn(t), velocity_fn(t)]).T

class TestDampedOscillator(unittest.TestCase):
    lst_args = [
        # (mass, k, alpha)
        (1.0, 1.0, 0.1),
        (1.0, 2.0, 0.2),
        (2.0, 1.0, 0.3)
    ]
    y0s = [
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 1.0]),
    ]
    
    def test_with_exact(self):
        t0, t1, dt = 0.0, 10.0, 0.01
        ts = jnp.arange(t0, t1, dt)
        term = ODETerm(damped_oscillator)
        solver = Dopri5()
        saveat = SaveAt(ts=ts)
        for args in self.lst_args:
            for y0 in self.y0s:
                with self.subTest(args=args, y0=y0):
                    term = ODETerm(damped_oscillator)
                    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
                    sol_exact = exact_sol(ts, y0, args)
                    self.assertTrue(jnp.allclose(sol.ys, sol_exact))

if __name__ == '__main__':
    unittest.main()
