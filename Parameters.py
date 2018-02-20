import numpy as np

### Parameters essential
test_case = 12
on_a_global_sphere = False
use_gpu = False

### Parameters secondary
# Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
timestepping = 'RK4'

# Duration, time stepping size, saving interval
#dt = 1440.   #1440 for 480km
dt = 360.   #360 for NA818
nYears = 50./360
save_inter_days = 10

# Model configuraitons, boundary conditions
delVisc = 0.  # 80 for NA818
bottomDrag =  0. #5.e-8
no_flux_BC = True  # Should always be on
no_slip_BC = False
free_slip_BC = True

# Solver config
linear_solver = 'lu'      # lu, cg, cudaCG, cudaPCG, amg
err_tol = 1e-8
max_iter = 5000
#max_iter_dual = 25
#dual_init = 'interpolation'    # 'extrapolation', 'interpolation'

restart = False
restart_file = 'restart.nc'

# Size of the phyiscal domain
earth_radius = 6371000.0
Omega0 = 7.292e-5

gravity = 9.81

# Forcing

# IO files
output_file = 'output.nc'

nTimeSteps = np.ceil(1.*86400*360/dt*nYears).astype('int')
save_interval = np.floor(1.*86400/dt*save_inter_days).astype('int')
if save_interval < 1:
    save_interval = 1

max_int = np.iinfo('int32').max

