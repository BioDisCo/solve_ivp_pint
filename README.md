# solve_ivp_pint

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![static analysis workflow](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/static-analysis.yaml/badge.svg)](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/static-analysis.yaml/)
[![test workflow](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/test.yaml/badge.svg)](https://github.com/BioDisCo/solve_ivp_pint/actions/workflows/test.yaml/)


This is the solve_ivp_pint library.

 This library allows you to use the “solve_ivp” ODE solver from the “scipy.integrate” library, while using the “Pint” library to assign units to its variables.
 
 This library's “solve_ivp” function has the same structure as the one in the “scipy.integrate” library:
 
 ```
 solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
 ```
 
 cf: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
 
 
