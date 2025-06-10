"""Simple example with a ball."""

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity, UnitRegistry

from solve_ivp_pint import solve_ivp

u = UnitRegistry()


# Define the ODE
def dxdt(t: Quantity, y: list[Quantity]) -> list:  # noqa: ARG001
    """
    dx/dt of a ball.

    y = [x, y, dx/dt, dy/dt]
    """
    vx = y[2]
    vy = y[3]
    dx_dt = vx
    dy_dt = vy
    dvx_dt = 0.0 * u.m / u.s**2
    dvy_dt = -9.81 * u.m / u.s**2
    return [dx_dt, dy_dt, dvx_dt, dvy_dt]


t0 = 0 * u.seconds  # initial time
tf = 3 * u.seconds  # final time

x_0: Quantity = 0.0 * u.m
y_0: Quantity = 2.0 * u.m  # we throw from 2m
vx_0: Quantity = 1.0 * u.m / u.s
vy_0: Quantity = 20.0 * u.m / u.s
y0 = [x_0, y_0, vx_0, vy_0]

t_eval = np.linspace(0, 3, 100) * u.s

# Solving
solution = solve_ivp(dxdt, [t0, tf], y0, t_eval=t_eval)

plt.title("Throwing a ball")
plt.plot([x.to(u.m).magnitude for x in solution.y[0]], [x.to(u.m).magnitude for x in solution.y[1]], "--")
plt.show()
