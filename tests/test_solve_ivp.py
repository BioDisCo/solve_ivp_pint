"""The tests."""

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from solve_ivp_pint import solve_ivp

u = UnitRegistry()

EPSILON = 1e-6


def test_solve_ivp() -> None:
    """Simple test."""

    # Define the ODE
    def dxdt(t: Quantity, y: Quantity) -> list:  # noqa: ARG001
        a: Quantity = 1 * u.seconds**-1
        b: Quantity = 2 * u.meters / u.seconds
        ret = 0 * u.meters / u.seconds - a * y[0] - b  # type: ignore
        return [ret]

    t0 = 0 * u.seconds  # initial time
    tf = 1 * u.seconds  # final time
    y0 = 0 * u.meters  # initial condition

    # Solving
    solution = solve_ivp(dxdt, [t0, tf], [y0])

    # Verifications
    assert solution.success, "Solving failed"
    assert len(solution.t) > 0, "Solution does not contain any time"
    assert len(solution.y[0]) > 0, "Solution does not contain any y value"


def test_linear() -> None:
    """Linear model."""

    # Define the ODE
    def dxdt(t: Quantity, y: Quantity) -> list:  # noqa: ARG001
        return [0.1 * u.m / u.s]

    t0 = 0 * u.seconds  # initial time
    tf = 10 * u.seconds  # final time
    y0 = 0 * u.meters  # initial condition

    # Solving
    solution = solve_ivp(dxdt, [t0, tf], [y0])

    assert solution.t[-1] == tf
    assert solution.y[0][-1] == 1 * u.m


def test_linear_teval_unit_outside() -> None:
    """Linear model with t_eval parameter, where the sequence has a unit."""

    # Define the ODE
    def dxdt(t: Quantity, y: Quantity) -> list:  # noqa: ARG001
        return [0.1 * u.m / u.s]

    t0 = 0 * u.seconds  # initial time
    tf = 10 * u.seconds  # final time
    y0 = 0 * u.meters  # initial condition

    # Solving
    solution = solve_ivp(dxdt, [t0, tf], [y0], t_eval=np.linspace(0, 10, 100) * u.s)

    assert solution.t[-1] == tf
    assert solution.y[0][-1].to("meter").magnitude == pytest.approx(1)


def test_linear_teval_unit_inside() -> None:
    """Linear model with t_eval parameter where each entry has a unit."""

    # Define the ODE
    def dxdt(t: Quantity, y: Quantity) -> list:  # noqa: ARG001
        return [0.1 * u.m / u.s]

    t0 = 0 * u.seconds  # initial time
    tf = 10 * u.seconds  # final time
    y0 = 0 * u.meters  # initial condition

    # Solving
    solution = solve_ivp(dxdt, [t0, tf], [y0], t_eval=[0 * u.s, 10 * u.s])

    assert solution.t[-1] == tf
    assert solution.y[0][-1].to("meter").magnitude == pytest.approx(1)


def test_ball() -> None:
    """Test fot throwing a ball."""

    # Define the ODE
    def dxdt(t: Quantity, y: list[Quantity]) -> list:  # noqa: ARG001
        """
        dy/dt of a ball.

        We assume the state y to be of the form [x, y, dx/dt, dy/dt]
        """
        vx = y[2]
        vy = y[3]
        dx_dt = vx
        dy_dt = vy
        dvx_dt = 0.0 * u.m / u.s**2
        dvy_dt = -9.81 * u.m / u.s**2
        return [dx_dt, dy_dt, dvx_dt, dvy_dt]

    t0: Quantity = 0 * u.seconds  # initial time
    tf: Quantity = 3 * u.seconds  # final time

    x_0: Quantity = 0.0 * u.m
    y_0: Quantity = 2.0 * u.m  # we throw from 2m
    vx_0: Quantity = 1.0 * u.m / u.s
    vy_0: Quantity = 20.0 * u.m / u.s
    y0 = [x_0, y_0, vx_0, vy_0]

    t_eval = np.linspace(0, 3, 100) * u.s

    # Solving
    solution = solve_ivp(dxdt, [t0, tf], y0, t_eval=t_eval)

    assert solution.t[-1] == tf
    assert np.abs(solution.y[0][-1].to(u.m).magnitude - 3.0) <= EPSILON  # x
    assert np.abs(solution.y[1][-1].to(u.m).magnitude - 17.855) <= EPSILON  # y
