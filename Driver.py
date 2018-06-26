import numpy as np
import Parameters as c
from ComputeEnvironment import ComputeEnvironment
from Grid import grid_data
from SWEs import state_data
from VectorCalculus import VectorCalculus
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import time
import os
from copy import deepcopy as deepcopy


max_int = np.iinfo('int32').max


    
        
def main( ):


    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

#    c = parameters()
    print("=========== Setting up the compute environment====================")
    env = ComputeEnvironment(c)

    print("=========== Init the grid object =================================")
    g = grid_data('grid.nc', c)

    print("===========Initializing the VectorCalculus object ================")
    vc = VectorCalculus(g, c, env)

    print("========== Initializing the State object =========================")
    s = state_data(g, c)

    ## Uncomment the following lines to perform tests
    print("========== Beginning tests =======================================")
#    from Testing import run_tests
#    run_tests(env, g, vc, c, s)
#    raise ValueError("Just for testing.")

    print("========== Setting the initial state of the model ================")
    s.initialization(g, vc, c)
#    raise ValueError("Just for testing.")
    
    print("========== Making a copy of the state object =====================")
    s_init = deepcopy(s)

    print("========== Declaring variables for holding statistics ============")
    # Compute energy and enstrophy
    kenergy = np.zeros(c.nTimeSteps+1)
    penergy = np.zeros(c.nTimeSteps+1)
    total_energy = np.zeros(c.nTimeSteps+1)
    mass = np.zeros(c.nTimeSteps+1)
    penstrophy = np.zeros(c.nTimeSteps+1)
    pv_max = np.zeros(c.nTimeSteps+1)
    pv_min = np.zeros(c.nTimeSteps+1)

    print("========== Computing some initial statistics =====================")
    h0 = np.mean(s_init.thickness[:])
    kenergy[0] = np.sum(s.thickness[:]*s.kinetic_energy[:]*g.areaCell[:])
    penergy[0] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
    total_energy[0] = kenergy[0] + penergy[0]
    mass[0] = np.sum(s.thickness[:] * g.areaCell[:])
    penstrophy[0] = 0.5 * np.sum(g.areaCell[:] * s.thickness * s.pv_cell[:]**2)
    pv_max[0] = np.max(s.pv_cell)
    pv_min[0] = np.min(s.pv_cell)


    print(("Running test case \#%d" % c.test_case))
    print(("K-nergy, p-energy, t-energy, p-enstrophy, mass: %e, %e, %e, %e, %e" % (kenergy[0], penergy[0], total_energy[0], penstrophy[0], mass[0])))

    error1 = np.zeros((c.nTimeSteps+1, 3)); error1[0,:] = 0.
    error2 = np.zeros((c.nTimeSteps+1, 3)); error2[0,:] = 0.
    errorInf = np.zeros((c.nTimeSteps+1, 3)); errorInf[0,:] = 0.

    s.save(c, g, 0)

    # Entering the loop
    t0 = time.clock( )
    t0a = time.time( )
    s_pre = deepcopy(s)
    s_old = deepcopy(s)
    s_old1 = deepcopy(s)
    
    for iStep in range(c.nTimeSteps):

        print(("Doing step %d/%d " % (iStep, c.nTimeSteps)))

        if c.timestepping == 'RK4':
            timestepping_rk4_z_hex(s, s_pre, s_old, s_old1, g, vc, c)
        elif c.timestepping == 'E':
            timestepping_euler(s, g, vc, c)
        else:
            raise ValueError("Invalid choice for time stepping")

        # Compute energy and enstrophy
        kenergy[iStep+1] = np.sum(s.thickness[:]*s.kinetic_energy[:]*g.areaCell[:])
        penergy[iStep+1] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
        total_energy[iStep+1] = kenergy[iStep+1] + penergy[iStep+1]
        mass[iStep+1] = np.sum(s.thickness[:] * g.areaCell[:])
        penstrophy[iStep+1] = 0.5 * np.sum(g.areaCell[:] * s.thickness[:] * s.pv_cell[:]**2)
        pv_max[iStep+1] = np.max(s.pv_cell)
        pv_min[iStep+1] = np.min(s.pv_cell)
#        aVorticity_total[iStep+1] = np.sum(g.areaCell * s.eta[:])
        
        print(("K-nergy, p-energy, t-energy, p-enstrophy, mass: %.15e, %.15e, %.15e, %.15e, %.15e" % \
              (kenergy[iStep+1], penergy[iStep+1], total_energy[iStep+1], penstrophy[iStep+1], mass[iStep+1])))
        print("min thickness: %f" % np.min(s.thickness))

        if kenergy[iStep+1] != kenergy[iStep+1]:
            raise ValueError("Exceptions detected in energy. Stop now")
        
        if np.mod(iStep+1, c.save_interval) == 0:
            k = (iStep+1) / c.save_interval
            s.save(c,g,k)

        if c.test_case == 2:
            # For test case #2, compute the errors
            error1[iStep+1, 0] = np.sum(np.abs(s.thickness[:] - s_init.thickness[:])*g.areaCell[:]) / np.sum(np.abs(s_init.thickness[:])*g.areaCell[:])
            error1[iStep+1, 1] = np.sum(np.abs(s.vorticity[:] - s_init.vorticity[:])*g.areaCell[:]) / np.sum(np.abs(s_init.vorticity[:])*g.areaCell[:])
            error1[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:])) 

            error2[iStep+1, 0] = np.sqrt(np.sum((s.thickness[:] - s_init.thickness[:])**2*g.areaCell[:]))
            error2[iStep+1,0] /= np.sqrt(np.sum((s_init.thickness[:])**2*g.areaCell[:]))
            error2[iStep+1, 1] = np.sqrt(np.sum((s.vorticity[:] - s_init.vorticity[:])**2*g.areaCell[:]))
            error2[iStep+1,1] /= np.sqrt(np.sum((s_init.vorticity[:])**2*g.areaCell[:]))
            error2[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:])) 

            errorInf[iStep+1, 0] = np.max(np.abs(s.thickness[:] - s_init.thickness[:])) / np.max(np.abs(s_init.thickness[:]))
            errorInf[iStep+1, 1] = np.max(np.abs(s.vorticity[:] - s_init.vorticity[:])) / np.max(np.abs(s_init.vorticity[:]))
            errorInf[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:]))

        s_tmp = s_old1
        s_old1 = s_old
        s_old = s_pre
        s_pre = s
        s = s_tmp

    days = c.dt * np.arange(c.nTimeSteps+1) / 86400.
    t1 = time.clock( )
    t1a = time.time( )
    plt.close('all')

    plt.figure(0)
    plt.plot(days, kenergy, '--', label="Kinetic energy", hold=True)
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    #plt.ylim(2.5e17, 2.6e17)
    plt.legend(loc=1)
    plt.savefig('energy.png', format='PNG')

    plt.figure(6)
    plt.plot(days, penergy, '-.', label="Potential energy", hold=True)
    plt.plot(days, total_energy, '-', label="Total energy")
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    #plt.ylim(8.0e20,8.15e20)
    plt.legend(loc=1)
    plt.savefig('total-energy.png', format='PNG')
    
    plt.figure(1)
    plt.plot(days, penstrophy)
    plt.xlabel('Time (days)')
    plt.ylabel('Enstrophy')
    #plt.ylim(0.74, 0.78)
    plt.savefig('enstrophy.png', format='PNG')
    print(("Change in potential enstrophy = %.15e " % (penstrophy[-1] - penstrophy[0])))

    plt.figure(5)
    plt.plot(days, mass)
    plt.xlabel('Time (days)')
    plt.ylabel('Mass')
    #plt.ylim(1.175e18, 1.225e18)
    plt.savefig('mass.png', format='PNG')

    plt.figure(7)
    plt.plot(days, pv_max, '-.', hold=True, label='PV max')
    plt.plot(days, pv_min, '--', hold=True, label='PV min')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.ylim([-0.0001, 0.0001])
    plt.xlabel('Days')
    plt.ylabel('Max/Min potential vorticity')
    plt.legend(loc=1)
    plt.savefig('pv_max_min.png', format='PNG')
    plt.savefig('pv_max_min.pdf', format='PDF')

    #plt.figure(8)
    #plt.plot(Years, aVorticity_total)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ##plt.ylim([-0.0001, 0.0001])
    #plt.xlabel('Years')
    #plt.ylabel('Total absolute vorticity')
    #plt.savefig('aVort_total.png', format='PNG')
    #plt.savefig('aVort_total.pdf', format='PDF')
    
    if c.test_case == 2:
        plt.figure(2); 
        plt.plot(days, error1[:,0], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,0], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,0], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-h.png', format='PNG')

        plt.figure(3); 
        plt.plot(days, error1[:,1], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,1], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,1], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-vorticity.png', format='PNG')

        plt.figure(4); 
        plt.plot(days, error1[:,2], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,2], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,2], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Absolute error')
        plt.savefig('error-divergence.png', format='PNG')

        print("Final l2 errors for thickness, vorticity, and divergence:")
        print(("                    %e,        %e,     %e" % (error2[-1,0], error2[-1,1], error2[-1,2])))
        

    print(('CPU time used: %f seconds' % (t1-t0)))
    print(('Walltime used: %f seconds' % (t1a-t0a)))

        
if __name__ == '__main__':
    main( )


            
        
    
