# MOP
Mathematical Ocean Prototype

Required software packages: Numpy, Scipy, matplotlib, netcdf4-python, f2py, Fortran compiler, pyamg

Optional packages: cupy (>= 8.0), AMGX, pyAMGX, cudatoolkit

Builing and execution procedure:
1. Build the Fortran module swe_comp

   $ bash build_fortran_module.sh
2. Place a copy of, or a link to, the grid file in the root directory of MOP
3. Edit the parameters in Parameter.py
4. Run the model

   $ ipython Driver.py

# Setup of a development and execution environment
1. Fortran compiler
2. (CUDA) Nvidia GPU driver
3. (CUDA) Compatible cudatoolkit
4. A Python3 environment
5. Python packages: numpy, scipy, matplotlib, netcdf4-python, f2py, pyamg
6. (CUDA) CUPY compatible with cudatoolkit
7. (CUDA) [AMGX](https://github.com/NVIDIA/AMGX)
  ```bash
mkdir build
cd build
cmake ../ -DCMAKE_NO_MPI=True
make -j8 all
```
8. (CUDA) [pyamgx](https://github.com/shwina/pyamgx)

Steps marked by CUDA are only necessary if the model is to run on Nvidia graphics cards using CUDA.


 


