from pint import UnitRegistry

from solve_ivp_pint import solve_ivp

ureg = UnitRegistry()


def test_solve_ivp():
    # Define the ODE
    def equation(t, y):  # noqa: ARG001
        a = 1 * ureg.seconds**-1
        b = 2 * ureg.meters / ureg.seconds
        sol = 0 - a * y[0] - b
        return [sol]

    t0 = 0 * ureg.seconds  # initial time
    tf = 1 * ureg.seconds  # final time
    y0 = 0 * ureg.meters  # initial condition

    # Résolution
    solution = solve_ivp(equation, [t0, tf], [y0])

    # Vérifications
    assert solution.success, "Solving failed"
    assert len(solution.t) > 0, "Solution do not contain any time"
    assert len(solution.y[0]) > 0, "Solution do not contain any y value"

    #Ajouter les badges dans Readme sur Github
