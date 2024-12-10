from pint import UnitRegistry

from solve_ivp_pint import solve_ivp

ureg = UnitRegistry()


def test_solve_ivp():
    # Définir l'équation différentielle
    def equation(t, y):  # noqa: ARG001
        a = 1 * ureg.seconds**-1
        b = 2 * ureg.meters / ureg.seconds
        sol = 0 - a * y[0] - b
        return [sol]

    t0 = 0 * ureg.seconds  # Temps initial
    tf = 1 * ureg.seconds  # Temps final
    y0 = 0 * ureg.meters  # Condition initiale

    # Résolution
    solution = solve_ivp(equation, [t0, tf], [y0])

    # Vérifications
    assert solution.success, "La résolution a échoué."
    assert len(solution.t) > 0, "La solution ne contient pas de points temporels."
    assert len(solution.y[0]) > 0, "La solution ne contient pas de valeurs pour y."
