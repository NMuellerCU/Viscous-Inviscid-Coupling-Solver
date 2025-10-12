import numpy as np
import matplotlib as mpl
import warnings
from typing import Literal


# 4 spacing options available, cosine bias is split into front and back for organization
Spacing = Literal["linear", "cosine", "front", "back"]

"""
            spacing options:
"""
# we want more points near LE and TE, so we use cosine spacing to achieve this
def cosine_spacing(N):
    beta = np.linspace(0, np.pi, N)
    x = 0.5*(1-np.cos(beta))
    return x

# evenly spaces points across airfoil
def linear_spacing(N):
    x = np.linspace(0, 1, N)
    return x
# if we want more points near LE   OR   TE, we use this
# bias < 0 == LE bias  bias > 0 == TE bias   bias = 0 == standard cosine bias
# Strength = 1 == default   strength > 1 == stronger
def cosine_spacing_bias(N, bias=0, strength=1):
    beta = np.linspace(0.0, np.pi, N)
    x = 0.5*(1-np.cos(beta))

    strength = max(1.0, float(strength))
    if bias < 0:
        x = x**strength # this compresses near 0 and relaxes near 1
    elif bias > 0:
        x = 1.0 - (1.0 - x)**strength  # this compresses near 1 and relaxes near 0

    # guarantee exact endpoints
    x[0] = 0.0
    x[-1] = 1.0
    return x


# function made to organize the spacing options
def calc_x(N, bias: float =0.0, strength:float = 1.0, spacing_option="consine"):
    if spacing_option == "cosine":
        if bias != 0 | strength != 1:
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for unbiased cosine spacing.",
                UserWarning,
                stacklevel=2
            )
        return cosine_spacing(N)
    elif spacing_option == "linear":
        if bias != 0 | strength != 1:
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for linear spacing.",
                UserWarning,
                stacklevel=2
            )
        return linear_spacing(N)
    elif spacing_option == "cosine_bias":
        return cosine_spacing_bias(N, bias, strength)


'''
naca4series: creates naca 4 series airfoil array values
m: max camber
p: location of max camber 
t: thickness 
n: number of points on surface
 '''
def naca4series(m: float, p: float, t: float, N: int, closed_te: bool = True, bias: float =0, strength: float =1, spacing_option="cosine") -> np.ndarray:

    x = calc_x(N, bias, strength, spacing_option)

    ''' 
    thickness distribution for naca 4 digit series is given by formula: 
    https://en.wikipedia.org/wiki/NACA_airfoil
    y = (t/c) * (A sqrt(x) - Bx - Cx^2 + Dx^3 - Ex^4)
    y: half thickness
    t/c: max thickness to chord ratio, 
    A: 0.2969
    B: 0.1260
    C: 0.3516
    D: 0.2843
    E: 0.1015 OR for closed_te 0.1036
    '''
    A = 0.2969
    B = 0.1260
    C = 0.3516
    D = 0.2843
    E = 0.1015
    if closed_te:
        E = 1.036
    y = 5 * t * ( A * np.sqrt(x) - B*x - C*x**2 + D*x**3 - E*x**4)

    '''
    find the mean camber line:
    y_c = m/p^2 ( 2px - x^2)  for 0 <= x <= p
        = m/(1-p)^2 ((1-2p) + 2px -x^2)  for p <= x <= 1
    use np.where: numpy.where(condition, [x, y, ]/)
    https://numpy.org/doc/2.3/reference/generated/numpy.where.html
    https://en.wikipedia.org/wiki/NACA_airfoil
    '''
    y_c = np.where(x < p,
                   m/p**2 *(2*p*x - x**2),
                   m/(1-p)**2 *((1-2*p) + 2*p*x - x**2)
                    )
    #     derivative of y_c
    dy_cdx = np.where( x < p,
                       2*m/p**2 * (p-x),
                       2*m/(1-p)**2 * (p-x)
                        )
    # theta
    theta = np.arctan2(dy_cdx, dy_cdx)












