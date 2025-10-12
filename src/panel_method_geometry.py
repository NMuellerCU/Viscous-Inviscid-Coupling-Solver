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
def calc_x(N, bias: float =0.0, strength:float = 1.0, spacing_option="cosine"):
    if spacing_option == "cosine":
        if (bias != 0) or (strength != 1):
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for unbiased cosine spacing.",
                UserWarning,
                stacklevel=2
            )
        return cosine_spacing(N)
    elif spacing_option == "linear":
        if (bias != 0) or (strength != 1):
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for linear spacing.",
                UserWarning,
                stacklevel=2
            )
        return linear_spacing(N)
    elif spacing_option == "cosine_bias":
        if strength < 1:
            warnings.warn(
                f"strength ({strength}) is less than 1 and is ignored.",
            UserWarning,
            stacklevel = 2
            )
        return cosine_spacing_bias(N, bias, strength)
    else:
        raise ValueError("spacing_option must be cosine, linear, or cosine_bias")



'''
x calculations for circle
'''
def CIRC_linear_spacing(N):
#     uniform arc lengths for circle
#   theta instead of x
    return np.linspace(0.0, 2*np.pi, N, endpoint=False)

def CIRC_cosine_spacing(N):
#    cosine spacing, made at 0,1 and scaled to 0,2pi
#   theta instead of x
    theta = 0.5*(1.0 - np.cos(np.linspace(0.0, 2*np.pi, N, endpoint=False)))
    return 2*np.pi * theta
def CIRC_cosine_spacing_bias(N, bias = 0.0, strength = 1):
    theta = 0.5 * (1.0 - np.cos(np.linspace(0.0, 2 * np.pi, N, endpoint=False)))
    strength = max(1.0, float(strength))
    if bias < 0:
        theta = theta**strength
    elif bias > 0:
        theta = 1.0 - (1.0 - theta)**strength
    return 2*np.pi * theta

def CIRC_calc_theta(N, bias: float =0.0, strength:float = 1.0, spacing_option="cosine"):
    if spacing_option == "cosine":
        if (bias != 0) or (strength != 1):
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for unbiased cosine spacing.",
                UserWarning,
                stacklevel=2
            )
        return CIRC_cosine_spacing(N)
    elif spacing_option == "linear":
        if (bias != 0) or (strength != 1):
            warnings.warn(
                f"bias ({bias}) and strength ({strength}) are ignored for linear spacing.",
                UserWarning,
                stacklevel=2
            )
        return CIRC_linear_spacing(N)
    elif spacing_option == "cosine_bias":
        if strength < 1:
            warnings.warn(
                f"strength ({strength}) is less than 1 and is ignored",
            UserWarning,
            stacklevel = 2
            )
        return CIRC_cosine_spacing_bias(N, bias, strength)

    else:
        raise ValueError("spacing_option must be cosine, linear, or cosine_bias")



'''
Verifications:
'''

def remove_duplicate(xy_all):
    if np.linalg.norm(xy_all[0] - xy_all[-1]) < 1e-10:
       xy_all = xy_all[:-1]
    return xy_all
#  check_clockwise: checks if the 2d array stack is clockwise, if not return false
def check_clockwise(xy_all):
    #  uses shoelace formula, if area A > 0 CCW if A < 0 CW
    #  https://en.wikipedia.org/wiki/Shoelace_formula
    # Note: this function is O(n) complexity, technically a similar numpy function is faster but also ~O(n), this is more readable so im leaving this
    s = 0
    n = len(xy_all)
    for i in range(n):
        x1, y1 = xy_all[i]
        x2, y2 = xy_all[(i+1) % n]
        s+= x1*y2 - x2*y1
    return  s < 0

#  make clockwise: checks if 2d array stack is clockwise, and if not make clockwise
def make_clockwise(xy_all):
    if not(check_clockwise(xy_all)):
          return xy_all[::-1] # just reverses the order xy_all
    else:
        return xy_all
'''
naca4series: creates naca 4 series airfoil array values
m: max camber
p: location of max camber 
t: thickness 
n: number of points on surface
 '''
def naca4series(m: float = 0.02, p: float = 0.4, t: float = 0.12, N: int = 100, closed_te: bool = True, bias: float =0, strength: float =1, spacing_option="cosine") -> np.ndarray:

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
        E = .1036
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
    theta = np.arctan(dy_cdx, dy_cdx)

    x_u = x - y*np.sin(theta)
    x_l = x + y*np.sin(theta)
    y_u = y_c + y*np.cos(theta)
    y_l = y_c - y * np.cos(theta)

    x_l_reverse = x_l[::-1]
    y_l_reverse = y_l[::-1]

    #  outputs organized in ccw order
    x_all = np.concatenate((x_u, x_l_reverse))
    y_all = np.concatenate((y_u, y_l_reverse))
    xy_all = np.column_stack((x_all, y_all))
    # cleanup makes sure the coordinate points are clockwise, and removes duplicate values at LE
    xy_all = make_clockwise(xy_all)
    if(closed_te):
        xy_all = remove_duplicate(xy_all)
    return xy_all

'''
naca4series: creates naca 4 series airfoil array values
r: radius
 '''
def circle( N: int = 100, bias: float =0, strength: float =1, spacing_option="cosine") -> np.ndarray:
    x = CIRC_calc_theta(N, bias, strength, spacing_option)
    x_u = x
    x_l = x[::-1]
    y_u = np.sqrt(1 - x**2)
    y_l = (-y_u)[::-1]

    x_all = np.concatenate((x_u, x_l))
    y_all = np.concatenate((y_u, y_l))
    xy_all = np.column_stack((x_all, y_all))
    xy_all = make_clockwise(xy_all)
    return xy_all


# def dynamic_triangle(theta: float = 60, phi: float = 0) -> np.ndarray:

# def polygon (P: int = 4  phi: float = 0, N: int = 100, bias: float =0, strength: float =1, spacing_option="cosine") -> np.ndarray:
#





