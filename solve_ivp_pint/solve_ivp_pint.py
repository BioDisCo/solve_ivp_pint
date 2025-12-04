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
    elif isinstance(t_span0, (list, tuple)):  # t_span0 is a tuple or a list
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


def solve_ivp(  # noqa: PLR0913, C901, PLR0912
    fun: Callable,
    t_span: list[Quantity] | tuple[Quantity],
    y0: list[Quantity] | tuple[Quantity],
    *,
    method: str = "RK45",
    t_eval: Quantity | list[Quantity] | None = None,
    dense_output: bool = False,
    events: Callable | list[Callable] | None = None,
    vectorized: bool = False,
    args: tuple | None = None,
    **options,  # noqa: ANN003
) -> OptimizeResult:
    """A solve_ivp function with pint units."""
    # Check of t_span's type
    if not isinstance(t_span, (list, tuple)):
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
    if t_eval is not None and isinstance(t_eval, (list, tuple)):
        # Check that each element has an attribute '_REGISTRY'
        for i, t in enumerate(t_eval):
            if not hasattr(t, "_REGISTRY"):
                msg = f"The element t_eval[{i}] ({t}) does not have a '_REGISTRY' attribute. Ensure it has units."
                raise TypeError(msg)

        # Verification of the compatibility between t_eval & t_span
        try:
            # Check the compatibility between t_eval & t_span
            if not all(item.check(t_span_units) for item in t_eval):  # type: ignore
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
            # Check the compatibility between t_eval & t_span
            if not t_eval.check(t_span_units):
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
