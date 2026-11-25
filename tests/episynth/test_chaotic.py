"""Numerical checks for the two integrators used by Chaotic."""

from collections import deque

import numpy as np
from scipy.integrate import solve_ivp

from episynth.generators.chaotic import Chaotic


def test_lorenz_matches_reference_with_fourth_order_convergence():
    def rhs(_time, state, sigma, rho, beta):
        x, y, z = state
        return [sigma * (y - x), rho * x - y - x * z, x * y - beta * z]

    parameters = np.array([[10.0, 28.0, 8 / 3], [11.0, 25.0, 3.25]])
    initial_states = np.array([[0.7, 0.2, -0.3], [-0.4, 0.8, 0.1]])
    duration = 0.2
    references = []
    for state, args in zip(initial_states, parameters, strict=True):
        solution = solve_ivp(
            rhs,
            (0, duration),
            state,
            args=args,
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
            dense_output=True,
        )
        assert solution.success, solution.message
        references.append(solution.sol)

    errors = []
    for steps in (10, 20, 40):
        times = np.linspace(0, duration, steps + 1)
        actual = Chaotic._lorenz(
            *parameters.T, duration / steps, len(times), initial_states.T
        )
        expected = np.stack([reference(times).T for reference in references], axis=1)
        errors.append(np.abs(actual - expected).max(axis=(0, 2)))

    assert np.all(errors[0] < 5e-4)
    assert np.all(errors[-1] < 2e-6)
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    # RK4 has global error O(dt**4), so halving dt should reduce it by ~16.
    np.testing.assert_allclose(ratios, 16, rtol=0.25)


def test_mackey_glass_matches_analytic_first_interval_with_euler_convergence():
    delay = 16
    initial = 0.9
    beta = 0.2
    gamma = 0.1
    exponent = 10
    # Constant history makes the feedback constant over the first delay.
    equilibrium = beta * initial / (gamma * (1 + initial**exponent))
    errors = []
    for dt in (2.0, 1.0, 0.5):
        steps = int(delay / dt)
        params = {
            "dt": dt,
            "beta": beta,
            "gamma": gamma,
            "n": exponent,
            "hist": steps + 1,
            "history": np.full(steps + 1, initial),
        }
        actual = Chaotic._mackey_glass([params], steps)[0]
        times = dt * np.arange(1, steps + 1)
        expected = equilibrium + (initial - equilibrium) * np.exp(-gamma * times)
        errors.append(np.max(np.abs(actual - expected)))

    assert errors[0] < 0.02
    assert errors[-1] < 0.005
    # Forward Euler has global error O(dt).
    np.testing.assert_allclose(np.asarray(errors[:-1]) / errors[1:], 2, rtol=0.15)


def test_mackey_glass_reuses_computed_values_after_the_delay():
    cases = [
        (2.0, 0.19, 0.08, 8, np.linspace(0.81, 0.98, 9)),
        (0.75, 0.23, 0.12, 12, np.linspace(0.97, 0.83, 25)),
    ]
    steps = 45
    params, expected = [], []
    for dt, beta, gamma, exponent, history in cases:
        params.append(
            {
                "dt": dt,
                "beta": beta,
                "gamma": gamma,
                "n": exponent,
                "hist": len(history),
                "history": history,
            }
        )
        value = history[-1]
        delay_line = deque(history[:-1])
        row = []
        for _ in range(steps):
            delayed = delay_line.popleft()
            delay_line.append(value)
            value += dt * (beta * delayed / (1 + delayed**exponent) - gamma * value)
            row.append(value)
        expected.append(row)

    actual = Chaotic._mackey_glass(params, steps)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
