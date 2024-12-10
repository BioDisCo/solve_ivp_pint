import pint
import scipy.integrate


def factory(model, t_span0, x_0, ureg):
    # Delete t_span0 and x_0 units (if any)
    x0_no_units = [item.magnitude for item in x_0]

    # Do deal with t_span0
    if hasattr(t_span0, "magnitude"):  # t_span0 is a unique quantity
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
    def f_no_units(t, x, *args):
        # Use the captured x_0 and t_span0
        x_units = [val * ureg.Unit(str(ref.units)) for val, ref in zip(x, x_0)]

        # Calculate derivatives
        dxdt_with_units = model(t, x_units, *args)

        return [
            term.to(ref.units / t_span_units).magnitude if not term.dimensionless else term.magnitude
            for term, ref in zip(dxdt_with_units, x_0)
        ]

    return f_no_units, x0_no_units, t_span_no_units


def solve_ivp(
    fun,
    t_span,
    y0,
    *,
    method="RK45",
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=None,
    **options,
):
    # Check of t_span's type
    if not isinstance(t_span, (list, tuple)):
        msg = f"Expected t_span to be of type list or tuple, but got {type(t_span).__name__}"
        raise TypeError(msg)
    # Check of the length
    nb_list = 2
    if len(t_span) != nb_list:
        msg = f"Expected t_span to contain exactly two elements, but got {len(t_span)}"
        raise ValueError(msg)

    # Check that each t_span's element has an attribut '_REGISTRY'
    for i, t in enumerate(t_span):
        if not hasattr(t, "_REGISTRY"):
            msg = f"The element t_span[{i}] ({t}) does not have a '_REGISTRY' attribute. Ensure it has units."
            raise TypeError(msg)

    retrieved_ureg = t_span[0]._REGISTRY  # noqa: SLF001
    # Verification of "options" that are not supported yet
    if options:  # If the dictionnary is not empty
        msg = "The function has not yet been implemented for the additional options provided: {}".format(
            ", ".join(options.keys())
        )
        raise NotImplementedError(msg)

    f_no_units, x0_no_units, t_span_no_units = factory(fun, t_span, y0, retrieved_ureg)

    # Management of t_eval: check if non None and that with t_span they have the same units (otherwise conversion), and then conversion without units
    if t_eval is not None and hasattr(t_eval, "dimensionality") and t_eval.dimensionality:
        # To get the units of t_span
        t_span_unit = t_span[0].units
        # Verification of the compatibility between t_eval & t_span
        try:
            # Check the compatibility between t_eval & t_span
            if not t_eval.check(t_span_unit):
                # Convertion of t_eval to have the same units as t_span
                t_eval = t_eval.to(t_span_unit)
        except pint.errors.DimensionalityError as e:
            # Will give an explicit pint error if the conversion fails
            msg = f"Failed to convert units of t_eval to match t_span. Error: {e}, please check the unit of t_eval, it should be the same as t_span"
            raise ValueError(msg) from e

        t_eval = t_eval.magnitude  # Convert to values without units

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

    return solution_sys
