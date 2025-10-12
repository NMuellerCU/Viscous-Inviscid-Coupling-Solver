import matplotlib.pyplot as mpl
from panel_method_geometry import naca4series


def main():

    naca2412 = naca4series(0.02,0.4,0.12,100)
    N = 4
    assert N >= 3
    if N % 2 == 0:
        N += 1  # enforce odd
    mpl.figure(figsize=(6,3))
    mpl.plot(naca2412[:,0], naca2412[:,1], 'k.-', lw=1, ms=2, label='Naca2412')
    mpl.axis('equal'); mpl.grid(True); mpl.legend(); mpl.title('Panels')
    mpl.show()


if __name__ == "__main__":
    main()


