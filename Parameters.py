import numpy as np

### Parameters essential
test_case = 2
use_gpu = False
use_gpu2 = True
performing_test = False

do_restart = False
restart_file = 'restart.nc'
save_restart_init = False

### Parameters secondary
# Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
timestepping = 'RK4'

# Choose energy conserving or energy-enstrophy conserving schemes
conserve_enstrophy = True     # False for energy-conserving only; True for both energy and enstrophy conserving 

# Duration, time stepping size, saving interval
dt = 1440.   #1440 for 480km
#dt = 90.   #360 for NA818
nYears = 5./360
save_inter_days = 1.

# Model configuraitons, boundary conditions
delVisc = 0.  # 80 for NA818
bottomDrag =  0. #5.e-8
no_flux_BC = True  # Should always be on
no_slip_BC = False
free_slip_BC = False

# Solver config
linear_solver = 'amgx'      # lu, cg, cudaCG, cudaPCG, amg, amgx
err_tol = 5e-8
max_iters = 1000
print_stats = 0             # 1 for True, 0 for False

# Size of the phyiscal domain
on_a_sphere = None
on_a_global_sphere = None
sphere_radius = 6371000.0
Omega0 = 7.292e-5

gravity = 9.80616

# Forcing

# IO files
output_file = 'output.nc'

nTimeSteps = np.ceil(1.*86400*360/dt*nYears).astype('int')
save_interval = np.floor(1.*86400/dt*save_inter_days).astype('int')
if save_interval < 1:
    save_interval = 1

max_int = np.iinfo('int32').max

