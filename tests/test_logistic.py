import unittest
from codes.logistic import logistic

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.typing import ArrayLike
from diffrax import ODETerm, diffeqsolve, Dopri5, SaveAt

def exact_sol(t: ArrayLike, N0:float, args: tuple[float, float]) -> ArrayLike:
    r, K = args
    return K / (1 + (K/N0-1)*jnp.exp(-r*t))

class TestLogistic(unittest.TestCase):
    lst_args = [
        # (r, K)
        (0.3, 100.0),
        (0.5, 100.0),
        (0.7, 50.0)
    ]
    y0s = [
        jnp.array(60.0),
        jnp.array(70.0),
        jnp.array(80.0),
    ]
    def test_stability(self):
        t0, t1, dt = 0.0, 100.0, 0.1
        term = ODETerm(logistic)
        solver = Dopri5()
        for args in self.lst_args:
            for y0 in self.y0s:
                with self.subTest(args=args, y0=y0):
                    term = ODETerm(logistic)
                    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args)
                    self.assertTrue(jnp.isclose(args[1], sol.ys[0]))

    def test_with_exact(self):
        t0, t1, dt = 0.0, 100.0, 0.1
        ts = jnp.arange(t0, t1, dt)
        term = ODETerm(logistic)
        solver = Dopri5()
        saveat = SaveAt(ts=ts)
        for args in self.lst_args:
            for y0 in self.y0s:
                with self.subTest(args=args, y0=y0):
                    term = ODETerm(logistic)
                    sol = diffeqsolve(term, solver, t0, t1, dt, y0, args, saveat=saveat)
                    sol_exact = exact_sol(ts, y0, args)
                    self.assertTrue(jnp.isclose(sol.ys, sol_exact).all())
    
if __name__ == '__main__':
    unittest.main()
