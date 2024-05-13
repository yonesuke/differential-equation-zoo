import unittest
from codes.harmonic_oscillator import harmonic_oscillator

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, vmap
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def exact_sol(t: ArrayLike, y0: ArrayLike, args: tuple[float, float]) -> ArrayLike:
    mass, k = args
    omega = jnp.sqrt(k / mass)
    position_fn = lambda t: y0[0] * jnp.cos(omega * t) + y0[1] / omega * jnp.sin(omega * t)
    velocity_fn = vmap(grad(position_fn))
    return jnp.vstack([position_fn(t), velocity_fn(t)]).T

class TestHarmonicOscillator(unittest.TestCase):
    lst_args = [
        # (mass, k)
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0)
    ]
    y0s = [
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 1.0]),
    ]
    
    def test_with_exact(self):
        t0, t1, dt = 0.0, 10.0, 0.01
        ts = jnp.arange(t0, t1, dt)
        term = ODETerm(harmonic_oscillator)
        solver = Dopri5()
        saveat = SaveAt(ts=ts)
        for args in self.lst_args:
            for y0 in self.y0s:
                with self.subTest(args=args, y0=y0):
                    term = ODETerm(harmonic_oscillator)
                    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
                    sol_exact = exact_sol(ts, y0, args)
                    self.assertTrue(jnp.allclose(sol.ys, sol_exact))

if __name__ == '__main__':
    unittest.main()
