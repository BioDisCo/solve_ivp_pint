"""The tests."""

from pint import Quantity, UnitRegistry

from solve_ivp_pint import solve_ivp

ureg = UnitRegistry()


def test_solve_ivp() -> None:
    """Simple test."""

    # Define the ODE
    def equation(t: Quantity, y: Quantity) -> list:  # noqa: ARG001
        a = 1 * ureg.seconds**-1
        b = 2 * ureg.meters / ureg.seconds
        sol = 0 - a * y[0] - b  # type: ignore
        return [sol]

    t0 = 0 * ureg.seconds  # initial time
    tf = 1 * ureg.seconds  # final time
    y0 = 0 * ureg.meters  # initial condition

    # Solving
    solution = solve_ivp(equation, [t0, tf], [y0])

    # Verifications
    assert solution.success, "Solving failed"
    assert len(solution.t) > 0, "Solution do not contain any time"
    assert len(solution.y[0]) > 0, "Solution do not contain any y value"
