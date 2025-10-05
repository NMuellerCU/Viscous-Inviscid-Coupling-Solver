# 2D-CFD-simulator

This project involves building a viscous inviscid coupling solver using a NACA 2412 airfoil. The goal is to simplify computation by seperating inviscid pressure effects and viscous skin friction effects and potentially seperation.

### Givens

- Airfoil Geometry: 2D NACA 2412 airfoil with nonzero lift at 0 AoA
- Inviscid outer layer, viscous boundary layer
- Steady incompressible 2D flow
- Re = 3e6, AoA = 0,2,4,6



### Goals
Determine Cp and Cf as a function of x along the chord of a NACA2412 airfoil at different pre-stall Angles of Attack.
Also to compare the results to the known Reynolds-Averaged Navier Stokes (RANS) values.
Perform sensitivity analysis to determine the effects of AoA and 