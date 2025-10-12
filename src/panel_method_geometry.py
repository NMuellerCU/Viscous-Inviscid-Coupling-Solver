import numpy as np
import matplotlib as mpl
import warnings
from typing import Literal


"""One-line summary in the imperative (what it does).

Optional longer description explaining key details, edge cases, or assumptions.

Args:
    param1: What this represents and any constraints (e.g., must be > 0).
    param2: What this represents and how it's used. Mention defaults if relevant.

Returns:
    What the function returns and what it means. If multiple return paths,
    describe each case briefly.

Raises:
    ValueError: When <condition>.
    TypeError: When <condition>.

Examples:
    >>> function_name(3, "hi")
    True
"""
"""One-line summary in the imperative (what it does).

Optional longer description explaining key details, edge cases, or assumptions.

Args:
    N: number of points used in the spacing 

Returns:
    What the function returns and what it means. If multiple return paths,
    describe each case briefly.

Raises:
    ValueError: When <condition>.
    TypeError: When <condition>.

Examples:
    >>> function_name(3, "hi")
    True
"""

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
def calc_x(N, bias=0, strength=1, spacing_option="consine"):
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


# creates naca 4 series airfoil
def naca4series(m: float, p: float, t: float, N: int, closed_te: bool = True, bias: float =0, strength: float =1, spacing_option="cosine") -> np.ndarray:

    x = calc_x(N, bias, strength, spacing_option)







