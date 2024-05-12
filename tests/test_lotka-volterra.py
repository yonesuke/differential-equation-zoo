import unittest
from codes.lotka_volterra import lotka_volterra

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

class TestLotkaVolterra(unittest.TestCase):
    lst_args = [
        # (alpha, beta, gamma, delta)
        (1.0, 0.1, 1.5, 0.75),
        (1.5, 0.1, 1.5, 0.75),
        (1.0, 0.2, 1.5, 0.75)
    ]
    y0s = [
        jnp.array([10.0, 5.0]),
        jnp.array([20.0, 10.0]),
        jnp.array([30.0, 15.0]),
    ]
    def test_constant_val(self):
        t0, t1, dt = 0.0, 100.0, 0.1
        ts = jnp.arange(t0, t1, dt)
        term = ODETerm(lotka_volterra)
        solver = Dopri5()
        saveat = SaveAt(ts=ts)
        for args in self.lst_args:
            for y0 in self.y0s:
                with self.subTest(args=args, y0=y0):
                    term = ODETerm(lotka_volterra)
                    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
                    alpha, beta, gamma, delta = args
                    conservation = gamma * sol.ys[:, 0] + beta * sol.ys[:, 1] - alpha * jnp.log(sol.ys[:, 1]) - delta * jnp.log(sol.ys[:, 0])
                    self.assertTrue(jnp.isclose(conservation, conservation[0] * jnp.ones_like(conservation)).all())

if __name__ == '__main__':
    unittest.main()
