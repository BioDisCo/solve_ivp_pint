# solve_ivp_pint: ODE solver with unit support using Pint 

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![static analysis workflow](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/static-analysis.yaml/badge.svg)](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/static-analysis.yaml/)
[![test workflow](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/test.yaml/badge.svg)](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/test.yaml/)

## Problem

If you love typing and units as we do, but need to resort to integration, this library may be for you.

The solve_ivp_pint library allows you to use the `solve_ivp` ODE solver from the `scipy.integrate` library, while using the `Pint` library to assign units to its variables.


## Install

Install via the pypi package `solve_ivp_pint`.
If you use pip, run the following shell command:

```shell
pip install solve_ivp_pint
```


## Use

This library's `solve_ivp` function has the same structure as the one in the `scipy.integrate` library:
 
```python
solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
```
 
For details on the (unitless) parameters see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
 
 
## Example

Let's model throwing a ball from 2 meters height with an initial speed of 1 m/s in the x-direction and 20 m/s in the y-direction.
Further, let's assume we do this on the earth with a gravitational acceleration of 9.81 in the opposite of the y-direction.
We can do this with solve_ivp_pint as:

```python
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity, UnitRegistry

from solve_ivp_pint import solve_ivp

u = UnitRegistry()


# Define the ODE
def dydt(t: Quantity, y: list[Quantity]) -> list:  # noqa: ARG001
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
```