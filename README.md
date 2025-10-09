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
Perform sensitivity analysis to determine the effects of panel count and BL steps.


### Notes and Progress

- The panel method, without diving too deep into computation, could be implemented into a solver where you give a set of
parameters for an airfoil and object and the ammount of panels to use and then you can compute a panel distribution for 
that object.

- For compressible flow, you can use the Prandtl-Glauert correction for compressibility using the Mach number, meaning
you only need to know the speed of sound (A lookup) and the freestream velocity, would be a simple correction to find
C_p for compressible.


**Panel Method Notes:**

- we define points but we need to keep them organized and ordered, with their ordering being 1 -> 2 -> ... -> n -> 1
- boundary points are the points that bound the panel, while the control point is the point at the
- \barS_a is the panel length
  - panel length is the pythagorean theorem
- always loop clockwise for direction of normal for velocity calculation




### Sources
**textbooks:**

Fundamentals of Aerodynamics, J. D. Anderson 6th Edition, McGraw

**youtube:**

Panel Method Geometry - JoshTheEngineer
https://www.youtube.com/watch?v=kIqxbd937PI

