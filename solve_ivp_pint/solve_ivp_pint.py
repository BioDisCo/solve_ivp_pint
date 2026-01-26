"""Integration library with units."""

from collections.abc import Callable, Sequence

import numpy as np
import pint
import scipy.integrate
from pint import Quantity, UnitRegistry
from scipy.optimize import OptimizeResult


def factory(
    model: Callable, t_span0: Quantity | list[Quantity] | tuple[Quantity], x_0: Sequence[Quantity], ureg: UnitRegistry
) -> tuple:
    """Create the unitless integration objects."""
    # Delete t_span0 and x_0 units (if any)
    x0_no_units = [item.magnitude for item in x_0]
    x0_units = [item.units for item in x_0]

    # Do deal with t_span0
    if isinstance(t_span0, Quantity):
        t_span_no_units = tuple(t_span0.magnitude)  # Convert to tuple
        t_span_units = t_span0.units  # Get the unit
    elif isinstance(t_span0, list | tuple):  # t_span0 is a tuple or a list
        t_span_no_units = tuple(item.magnitude if hasattr(item, "magnitude") else item for item in t_span0)
        # Check that the 2 elements have the same unit
        if all(hasattr(item, "units") for item in t_span0):
            t_span_units = t_span0[0].units  # Take the first element unit
            if not all(item.units == t_span_units for item in t_span0):
                msg = "All elements in t_span0 must have the same units."
                raise ValueError(msg)
        else:
            msg = "t_span0 elements must have units."
            raise ValueError(msg)
    else:
        msg = "t_span0 must be a tuple/list of quantities or a single quantity with units."
        raise TypeError(msg)

    # Defines f_no_units as a closure
    def f_no_units(t: np.number, x: np.ndarray | tuple[float], *args: tuple) -> list:
        # Use the captured x_0 and t_span0
        x_units = [val * ureg.Unit(str(ref.units)) for val, ref in zip(x, x_0, strict=False)]

        # Calculate derivatives
        dxdt_with_units = model(t, x_units, *args)

        return [
            term.to(ref.units / t_span_units).magnitude if not term.dimensionless else term.magnitude
            for term, ref in zip(dxdt_with_units, x_0, strict=False)
        ]

    return f_no_units, x0_no_units, t_span_no_units, t_span_units, x0_units


# By default atol = 1e-6 & rtol = 1e-3 - cf https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
def solve_ivp(  # noqa: C901, PLR0912, PLR0913, PLR0915
    fun: Callable,
    t_span: list[Quantity] | tuple[Quantity],
    y0: list[Quantity] | tuple[Quantity],
    *,
    method: str = "RK45",
    t_eval: Quantity | list[Quantity] | None = None,
    dense_output: bool = False,
    events: Callable | list[Callable] | None = None,
    vectorized: bool = False,
    atol: float | list | tuple | Quantity | list[Quantity] | tuple[Quantity] | None = 1e-6,
    rtol: float | list | tuple | None = 1e-3,
    args: tuple | None = None,
    **options,  # noqa: ANN003
) -> OptimizeResult:
    """A solve_ivp function with pint units."""
    # Check of t_span's type
    if not isinstance(t_span, list | tuple):
        msg = f"Expected t_span to be of type list or tuple, but got {type(t_span).__name__}"
        raise TypeError(msg)
    # Check of the length
    nb_list = 2
    if len(t_span) != nb_list:
        msg = f"Expected t_span to contain exactly two elements, but got {len(t_span)}"
        raise ValueError(msg)

    # Check that each t_span's element has an attribute '_REGISTRY'
    for i, t in enumerate(t_span):
        if not hasattr(t, "_REGISTRY"):
            msg = f"The element t_span[{i}] ({t}) does not have a '_REGISTRY' attribute. Ensure it has units."
            raise TypeError(msg)

    ureg = t_span[0]._REGISTRY  # noqa: SLF001
    # Verification of "options" that are not supported yet
    if options:  # If the dictionary is not empty
        msg = "The function has not yet been implemented for the additional options provided: {}".format(
            ", ".join(options.keys())
        )
        raise NotImplementedError(msg)

    f_no_units, x0_no_units, t_span_no_units, t_span_units, x0_units = factory(fun, t_span, y0, ureg)

    # Management of t_eval: check if non None and that with t_span they have the same units
    # (otherwise conversion), and then conversion without units

    # case: t_eval is not None and is a list or tuple of quantities with units
    if t_eval is not None and isinstance(t_eval, list | tuple):
        # Check that each element has an attribute '_REGISTRY'
        for i, t in enumerate(t_eval):
            if not hasattr(t, "_REGISTRY"):
                msg = f"The element t_eval[{i}] ({t}) does not have a '_REGISTRY' attribute. Ensure it has units."
                raise TypeError(msg)

        # Verification of the compatibility between t_eval & t_span
        try:
            # Conversion of t_eval to have the same units as t_span
            t_eval = [item.to(t_span_units) for item in t_eval]  # type: ignore
        except pint.errors.DimensionalityError as e:
            # Will give an explicit pint error if the conversion fails
            msg = (
                "Failed to convert units of t_eval to match t_span."
                f"Error: {e}, please check the unit of t_eval, it should be the same as t_span"
            )
            raise ValueError(msg) from e

        t_eval = [item.magnitude for item in t_eval]  # type: ignore # Convert to values without units

    elif t_eval is not None and hasattr(t_eval, "dimensionality") and t_eval.dimensionality:
        # Verification of the compatibility between t_eval & t_span
        try:
            # Conversion of t_eval to have the same units as t_span
            t_eval = t_eval.to(t_span_units)  # type: ignore
        except pint.errors.DimensionalityError as e:
            # Will give an explicit pint error if the conversion fails
            msg = (
                "Failed to convert units of t_eval to match t_span."
                f"Error: {e}, please check the unit of t_eval, it should be the same as t_span"
            )
            raise ValueError(msg) from e

        t_eval = t_eval.magnitude  # type: ignore # Convert to values without units

    # case: a_tol is not None and is a list/tuple of Quantity with units
    if atol is not None and isinstance(atol, (list | tuple)):
        # Check first if we have a list or tuple with units or not
        has_units = any(hasattr(item, "_REGISTRY") for item in atol)

        if has_units:
            # Check that each element has an attribute '_REGISTRY'
            for i, t in enumerate(atol):
                if not hasattr(t, "_REGISTRY"):
                    msg = f"The element atol[{i}] ({t}) does not have a '_REGISTRY' attribute. Ensure it has units."
                    raise TypeError(msg)

            # Check length compatibility
            if len(atol) != len(y0):
                msg = f"atol must have the same length as y0. Got {len(atol)} for atol vs {len(y0)} for y0."
                raise ValueError(msg)

            # Verification of the compatibility between atol & y0
            try:
                # Check the compatibility between atol & x0
                atol_converted = []
                for item, unit in zip(atol, x0_units, strict=True):
                    converted_item = item.to(unit)
                    atol_converted.append(converted_item)
                atol = atol_converted
            except pint.errors.DimensionalityError as e:
                # Will give an explicit pint error if the conversion fails
                msg = (
                    "Failed to convert units of atol to match y0."
                    f"Error: {e}, please check the unit of atol, it should be the same as y0"
                )
                raise ValueError(msg) from e

            atol = [item.magnitude for item in atol]  # type: ignore # Convert to values without units

        # If the list / tuple is without units, we check the length
        elif len(atol) != len(y0):
            msg = f"atol must have the same length as y0. Got {len(atol)} for atol vs {len(y0)} for y0."
            raise ValueError(msg)

    elif atol is not None and isinstance(atol, Quantity) and atol.dimensionality:
        # Check if all y0 components have compatible dimensions
        first_unit = x0_units[0]
        all_compatible = all(atol.check(unit) for unit in x0_units)

        if all_compatible is False:
            msg = (
                "When using a scalar atol with units, all components of y0 must have "
                "compatible dimensions. Your y0 has heterogeneous units: "
                f"{[str(unit) for unit in x0_units]}. "
                "Please provide either:\n"
                "  - A scalar atol without units (ex : atol = 1e-8)\n"
                f"  - A list/tuple of atol values with the same units as y0 (ex :  atol = [1e-8 * {first_unit}, ...])"
            )
            raise ValueError(msg)

        # Verification of the compatibility between atol & x0
        try:
            # Convert to a list with each element having the appropriate unit
            atol_converted = []
            for unit in x0_units:
                converted = atol.to(unit)
                atol_converted.append(converted.magnitude)
            atol = atol_converted
        except pint.errors.DimensionalityError as e:
            # Will give an explicit pint error if the conversion fails
            msg = (
                "Failed to convert units of atol to match y0."
                f"Error: {e}, please check the unit of atol, it should be the same as y0"
            )
            raise ValueError(msg) from e

        atol = atol.magnitude  # type: ignore # Convert to values without units

    # case rtol is not None and is a Quantity, check if it's without units
    # case rtol is not None and is a list or tuple, check the length and if it's without any unit
    if rtol is not None:
        if isinstance(rtol, Quantity) and not rtol.dimensionless:
            msg = "rtol must be dimensionless"
            raise ValueError(msg)

        if isinstance(rtol, list | tuple):
            if len(rtol) != len(y0):
                msg = f"rtol must have the same length as y0. Got {len(rtol)} for rtol vs {len(y0)} for y0."
                raise ValueError(msg)

            # Check that all elements are dimensionless
            for i, r in enumerate(rtol):
                if isinstance(r, Quantity) and not r.dimensionless:
                    msg = f"All elements of rtol must be dimensionless. Element rtol[{i}] has units: {r.units}"
                    raise ValueError(msg)

    # Calling 'solve_ivp' to solve ODEs
    solution_sys = scipy.integrate.solve_ivp(
        f_no_units,
        t_span_no_units,
        x0_no_units,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        atol=atol,
        rtol=rtol,
        args=args,
        **options,
    )

    # Checking for simulation errors
    if not solution_sys.success:
        msg = "The simulation failed to converge."
        raise RuntimeError(msg)

    # Add units back in to solution
    solution_sys.t = [time * t_span_units for time in solution_sys.t]
    solution_sys.y = [[val * unit for val in vals] for vals, unit in zip(solution_sys.y, x0_units, strict=False)]

    return solution_sys
