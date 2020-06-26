# MOP
Mathematical Ocean Prototype

Required software packages: Numpy, Scipy, matplotlib, netcdf4-python, f2py, Fortran compiler, pyamg

Optional packages: AMGX, pyAMGX, cudatoolkit

Builing and execution procedure:
1. Build the Fortran module swe_comp

   $ bash build_fortran_module.sh
2. Place a copy of, or a link to, the grid file in the root directory of MOP
3. Edit the parameters in Parameter.py
4. Run the model

   $ ipython Driver.py

# Installation
Here's a brief review of the steps that I needed to take to run the package on Ubuntu 18.04. (Everything is done using Anaconda) 
## Dependencies (CPU):
Create a file named ```environment.yml``` and populate it with this code:

    name: REU  #choose any name you like
    channels:
        - conda-forge
    dependencies:
        - python  #for latest python
        - ipython
        - numpy
        - scipy
        - matplotlib
        - fastscapelib-f2py
        - cython
        - hdf5
        - netcdf4
        - time

Now, you can simply follow these steps:

 1. Run: ```conda env create -f environment.yml```
 2. Run: ```bash build_fortran_module.sh```
 3. Now you must take a grid file (```grid.nc```) and place it into the base directory. 
 4. Run: ```ipython Driver.py```

## Dependencies (GPU):
(Note: this is slightly more involved)

You will need to first go through and install everything needed for the CPU portion.

### CudaToolkit (latest version):

 - Get the runfile from [Nvidia's](https://developer.nvidia.com/cuda-downloads) website.
 - (Optional: deactivate any Nvidia drivers and package managers. To do this, go to the Software Updater application. Then, select *Settings & Livepatch*. This should take you into a window with many tabs. Click the *Additional Drivers* tab. Select *Using Xorg X Server*. Finally, click *Apply Changes*.)
 - Disable the Nouveau drivers.
	 - Run: ```sudo touch /etc/modprobe.d/blacklist-nouveau.conf```
	 - In any editor modify this new file to contain (on separate lines): 
	 - ```blacklist nouveau```
	 - ```options nouveau modeset=0```
	 - Run: ```sudo update-initramfs -u```
 - For the next part, you will need to be in *runlevel 3* (text mode).
	 - Run: ```sudo init 3```
	 - Login to your profile
	 - Navigate to the location where you installed the runfile
	 - Run: ```sudo sh <your-filename>.run```
	 - Follow the instructions and reboot

### AMGX:

 - Download the AMGX [repository](https://github.com/NVIDIA/AMGX).
 - Run: 
	 - ```mkdir build```
	 - ```cd build```
	 - ```cmake ../```
	 - ```make -j16 all```

### pyamgx

 - Download the pyamgx [repository](https://github.com/shwina/pyamgx).
 - Run:
	 - ```export AMGX_DIR=/path/to/.../AMGX```
	 - ```export AMGX_BUILD_DIR=/path/to/.../build```
 - Run: ```pip install .```

### Test GPU Installation:
Modify the ```Parameters.py``` file by simply changing the existing line to:

 - ```33: linear_solver = 'amgx'```
 - Run: ```python Driver.py```

 


