# cqExperiments/nonlinearMaxwell
The following commands (with a shell in the folder cqExperiments, one level above the present level) recover the plots of the corresponding manuscript:

python3 nonlinearMaxwell/generate_grids.py 
python3 nonlinearMaxwell/create_densities.py
python3 nonlinearMaxwell/create_error_data.py 
python3 nonlinearMaxwell/density2pointevals.py

This computes the densities, the errors for the convergence plots and the field data for the images of Figure 3.

Finally, running the Matlabscript "plottingScripts/ErrorPlotsMain.m" yields the convergence plots and the script "plottingScripts/FramesPlotPub.m" yields the final Figure 3.


