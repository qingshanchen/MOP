import numpy as np

### Parameters essential
test_case = 5
use_gpu = False
performing_test = False

if use_gpu:
    import cupy as xp
else:
    import numpy as xp
vector_order = 'F'
    
#nLayers = 1
#rho_vec = xp.array([1000.]) # index 0 = top layer
#rho0 = 1000.

nLayers = 2
rho_vec = xp.array([1000.,1010.]) # index 0 = top layer
rho0 = 1000.

do_restart = True
restart_file = 'output-tc5-2layer-40962-day9-16.nc'
save_restart_init = False

### Parameters secondary
# Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
timestepping = 'RK4'

# Choose energy conserving or energy-enstrophy conserving schemes
conserve_enstrophy = True     # False for energy-conserving only; True for both energy and enstrophy conserving 

# Duration, time stepping size, saving interval
dt = 45.   #1440 for 480km
#dt = 90.   #360 for NA818
nYears = 34./360
save_inter_days = 1.

# Model configuraitons, boundary conditions
delVisc = 0.  # 80 for NA818
bottomDrag =  0. #5.e-8
GM_kappa = 0.
kappa = 1.e11 #2.e12
no_flux_BC = True  # Should always be on
no_slip_BC = False
free_slip_BC = False
sigma = 2.e200
min_thickness = 10. # Minimum layer thickness
power = 2 # Power of the artificial potential energy


# Solver config
linear_solver = 'lu'      # lu, cg, cudaCG, cudaPCG, amg, amgx
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

