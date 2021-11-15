# cqExperiments
Project implements a black box CQ solver, suitable for both linear and nonlinear problems. Main application are scattering problems by making use of the BEM - Library Bempp. The toolbox is capable of solving a general class of evolution equations, possibly nonlocal in time and space under the assumption of vanishing initial conditions. 

The folder structure is as follows:
- cqToolbox contains scripts regarding the Convolution quadrature method. The script "linearcq.py" implements a forward evaluation of the convolution quadrature method, cqStepper.py builds a class, which reduces an abstract time stepping problem to a sequence of forward evaluations, which are then computed by "linearcq.py".

- linearMaxwell contains simulation scripts for time-dependent Maxwell's equations with linear boundary conditions, overall a linear problem. The scripts make use of Bempp and the functionality provided by "linearcq.py" to solve these problems.

- nonlinearMaxwell contains simulation scripts for time-dependent Maxwell's equations with nonlinear boundary conditions, rendering the complete problem Semilinear. The scripts make use of Bempp and the functionality provided by newtonStepper.py, which builds upon cqStepper.py to solve the system in a marching-on-in time fashion.

- data collects the simulation data.

- plottingScripts collects a bunch of Matlab routines vor visualization.


