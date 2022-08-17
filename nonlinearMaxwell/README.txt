# cqExperiments/nonlinearMaxwell

The following package versions were used for the present experiments (Given through the output of nonlinearMaxwell/libversions.py):

Python version: (3, 5, 2, 'final', 0)
Bempp version : 3.3.4
Scipy version : 1.4.1
Numpy version : 1.18.5


The following commands (with a shell in the folder cqExperiments, one level above the present level) recover the plots of the corresponding manuscript:

python3 nonlinearMaxwell/generate_grids.py 
python3 nonlinearMaxwell/create_densities.py
python3 nonlinearMaxwell/create_error_data.py 
python3 nonlinearMaxwell/density2pointevals.py

This computes the densities, the errors for the convergence plots and the field data for the images of Figure 3.

Finally, running the Matlabscript "plottingScripts/ErrorPlotsMain.m" yields the convergence plots and the script "plottingScripts/FramesPlotPub.m" yields the final Figure 3.


