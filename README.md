# MOP
Mathematical Ocean Prototype

Required software packages: Numpy, Scipy, matplotlib, netcdf4-python, f2py, Fortran compiler
Optional packages: AMGX, pyAMGX, cudatoolkit

Builing and execution procedure:
1. Build the Fortran module swe_comp 
   $bash build_fortran_module.sh
2. Place a copy of, or a link to, the grid file in the root directory of MOP
3. Edit the parameters in Parameter.py
4. Run the model 
   $ipython Driver.py
