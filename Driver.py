import numpy as np
import Parameters as c
from Grid import grid_data
from State import state_data, timestepping_rk4_z_hex
from VectorCalculus import VectorCalculus
from Elliptic import EllipticCpl2
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import time
import os
from copy import deepcopy as deepcopy


max_int = np.iinfo('int32').max


def main( ):

    # load appropriate module for interacting with arrays on GPU / CPU
    if c.use_gpu:
        import cupy as xp
    else:
        import numpy as xp
    
    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

    print("=========== Init the grid object =================================")
    g = grid_data('grid.nc', c)

    print("===========Initializing the VectorCalculus object ================")
    vc = VectorCalculus(g, c)

    print("===========Initializing the Poisson object ================")
    poisson = EllipticCpl2(vc, g, c)

    print("========== Initializing the State object =========================")
    s = state_data(vc, g, c)

    ## Uncomment the following lines to perform tests
    if c.performing_test:
        print("========== Beginning tests =======================================")
        from Testing import run_tests
        run_tests(g, c, s, vc, poisson)
        raise ValueError("Just for testing.")

    print("========== Setting the initial state of the model ================")
    s.initialization(poisson, g, vc, c)

    
    print("========== Making a copy of the state object =====================")
    s_init = deepcopy(s)

    print("========== Declaring variables for holding statistics ============")
    # Compute energy and enstrophy
    kinetic_energy = np.zeros( c.nTimeSteps+1 )
    pot_energy = np.zeros( c.nTimeSteps+1)
    art1_energy = np.zeros(c.nTimeSteps+1)
    art2_energy = np.zeros(c.nTimeSteps+1)
    total_energy = np.zeros(c.nTimeSteps+1)
    total_energy2 = np.zeros(c.nTimeSteps+1)
    mass = np.zeros( (c.nTimeSteps+1, c.nLayers) )
    pot_enstrophy = np.zeros(c.nTimeSteps+1)
    total_vorticity = np.zeros((c.nTimeSteps+1, c.nLayers) )
    pv_max = np.zeros( (c.nTimeSteps+1,c.nLayers) )
    pv_min = np.zeros( (c.nTimeSteps+1,c.nLayers) )
    avg_divergence = np.zeros( (c.nTimeSteps+1,c.nLayers) )
    timeNow = np.zeros(c.nTimeSteps+1)

    print("========== Computing some initial statistics =====================")
    if c.use_gpu:
        kinetic_energy[0] = s.kinetic_energy.get()
        pot_energy[0] = s.pot_energy.get()
        art1_energy[0] = s.art1_energy.get()
        art2_energy[0] = s.art2_energy.get()
        total_energy[0] = kinetic_energy[0] + pot_energy[0]
        total_energy2[0] = total_energy[0] + art1_energy[0] + art2_energy[0]
        mass[0,:] = xp.sum(s.thickness * g.areaCell, axis=0).get()
        pot_enstrophy[0] = s.pot_enstrophy.get()
        total_vorticity[0,:] = xp.sum(s.pv_cell * s.thickness * g.areaCell, axis=0).get()
#        pv_max[0,:] = xp.max(s.pv_cell, axis=0).get()
#        pv_min[0,:] = xp.min(s.pv_cell, axis=0).get()
        timeNow[0] = s.time
    else:
        kinetic_energy[0] = s.kinetic_energy
        pot_energy[0] = s.pot_energy
        art1_energy[0] = s.art1_energy
        art2_energy[0] = s.art2_energy
        total_energy[0] = kinetic_energy[0] + pot_energy[0]
        total_energy2[0] = total_energy[0] + art1_energy[0] + art2_energy[0]
        mass[0,:] = xp.sum(s.thickness * g.areaCell, axis=0)
        pot_enstrophy[0] = s.pot_enstrophy
        total_vorticity[0,:] = xp.sum(s.pv_cell * s.thickness * g.areaCell, axis=0)
#        pv_max[0,:] = xp.max(s.pv_cell, axis=0)
#        pv_min[0,:] = xp.min(s.pv_cell, axis=0)
        timeNow[0] = s.time

    print(("Running test case \#%d" % c.test_case))
    print("Mass for each layer: ", mass[0,:])
    print("Total vorticity for each layer: ", total_vorticity[0,:])
#    print(("K-energy, p-energy, t-energy, p-enstrophy, mass: %.15e, %.15e, %.15e, %.15e" % (kinetic_energy[0], pot_energy[0], total_energy[0], pot_enstrophy[0])))
    print(("k-energy, p-energy, a1-energy, a2-energy, t-energy, t2_energy, p-enstrophy: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e" % \
           (kinetic_energy[0], pot_energy[0], art1_energy[0], art2_energy[0], total_energy[0], total_energy2[0], pot_enstrophy[0])))

    if c.test_case == 2 or c.test_case == 12:
        error1 = xp.zeros((c.nTimeSteps+1, 3, c.nLayers)); error1[0,:,:] = 0.
        error2 = xp.zeros((c.nTimeSteps+1, 3, c.nLayers)); error2[0,:,:] = 0.
        errorInf = xp.zeros((c.nTimeSteps+1, 3, c.nLayers)); errorInf[0,:,:] = 0.

    # Save the initial state when starting from function
    # or when starting from file and told to save the inital state
    if not c.do_restart:
        nc_num = 0
        s.save(c, g, nc_num)
    elif c.do_restart and c.save_restart_init:
        nc_num = 0
        s.save(c, g, nc_num)
    else:
        nc_num = -1

    # Entering the loop
    t0 = time.process_time( )
    t0a = time.time( )
    s_pre = deepcopy(s)
    s_old = deepcopy(s)
#    s_old1 = deepcopy(s)
    
    for iStep in range(c.nTimeSteps):

        print(("Doing step %d/%d " % (iStep+1, c.nTimeSteps)))

        if c.timestepping == 'RK4':
            timestepping_rk4_z_hex(s, s_pre, s_old, poisson, g, vc, c)
        elif c.timestepping == 'E':
            timestepping_euler(s, poisson, g, vc, c)
        else:
            raise ValueError("Invalid choice for time stepping")

        # Compute energy and enstrophy
        if c.use_gpu:
            kinetic_energy[iStep+1] = s.kinetic_energy.get()
            pot_energy[iStep+1] = s.pot_energy.get()
            art1_energy[iStep+1] = s.art1_energy.get()
            art2_energy[iStep+1] = s.art2_energy.get()
            total_energy[iStep+1] = kinetic_energy[iStep+1] + pot_energy[iStep+1]
            total_energy2[iStep+1] = total_energy[iStep+1] + art1_energy[iStep+1] + art2_energy[iStep+1]
            mass[iStep+1,:] = xp.sum(s.thickness * g.areaCell, axis=0).get()
            pot_enstrophy[iStep+1] = s.pot_enstrophy.get()
            total_vorticity[iStep+1,:] = xp.sum(s.pv_cell * s.thickness * g.areaCell, axis=0).get()
#            pv_max[iStep+1,:] = xp.max(s.pv_cell, axis=0).get()
#            pv_min[iStep+1,:] = xp.min(s.pv_cell, axis=0).get()
        else:
            kinetic_energy[iStep+1] = s.kinetic_energy
            pot_energy[iStep+1] = s.pot_energy
            art1_energy[iStep+1] = s.art1_energy
            art2_energy[iStep+1] = s.art2_energy
            total_energy[iStep+1] = kinetic_energy[iStep+1] + pot_energy[iStep+1]
            total_energy2[iStep+1] = total_energy[iStep+1] + art1_energy[iStep+1] + art2_energy[iStep+1]
            mass[iStep+1,:] = xp.sum(s.thickness * g.areaCell, axis=0)
            pot_enstrophy[iStep+1] = s.pot_enstrophy
            total_vorticity[iStep+1,:] = xp.sum(s.pv_cell * s.thickness * g.areaCell, axis=0)
#            pv_max[iStep+1,:] = xp.max(s.pv_cell, axis=0)
#            pv_min[iStep+1,:] = xp.min(s.pv_cell, axis=0)

        timeNow[iStep+1] = s.time

        print("Mass for each layer: ", mass[iStep+1,:])
        print("Total vorticity for each layer: ", total_vorticity[iStep+1,:])
        print(("k-energy, p-energy, a1-energy, a2-energy, t-energy, t2_energy, p-enstrophy: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e" % \
               (kinetic_energy[iStep+1], pot_energy[iStep+1], art1_energy[iStep+1], art2_energy[iStep+1], total_energy[iStep+1], total_energy2[iStep+1], pot_enstrophy[iStep+1])))
        for iLayer in range(c.nLayers):
            print("min and max thickness: %f, %f" % (xp.min(s.thickness[:,iLayer]), xp.max(s.thickness[:,iLayer])) )

        if np.mod(iStep+1, c.save_interval) == 0:
            nc_num += 1
            s.save(c, g, nc_num)

        if c.test_case == 2 or c.test_case == 12:
            s.compute_tc2_errors(iStep, s_init, error1, error2, errorInf, g)

#        s_tmp = s_old1
        s_tmp = s_old
        s_old = s_pre
        s_pre = s
        s = s_tmp

    days = timeNow[:] / 86400.
    t1 = time.process_time( )
    t1a = time.time( )

    if True: 
        plt.close('all')

        plt.figure(0)
        plt.plot(days, (total_energy-total_energy[0])/total_energy[0], '--')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Time (days)')
        plt.ylabel('Normalized changes in total energy')
        #plt.ylim(2.5e17, 2.6e17)
        plt.savefig('total_energy_change.png', format='PNG')
        print(("Change in total energy = %e " % (np.abs(total_energy[-1] - total_energy[0])/total_energy[0])))
        print(("Change in total energy2 = %e " % (np.abs(total_energy2[-1] - total_energy2[0])/total_energy2[0])))

        plt.figure(6)
        plt.plot(days, kinetic_energy, '--', label="Kinetic energy")
        plt.plot(days, pot_energy, '-.', label="Potential energy")
        plt.plot(days, art1_energy, '-.', label="Art. PE (polynm)")
        plt.plot(days, art2_energy, '-.', label="Art. PE (Grad)")
        plt.plot(days, total_energy, 'r-', label="K. + pot. energy")
        plt.plot(days, total_energy2, 'k-', label="K. + pot.+ art. pot. energy")
        plt.xlabel('Time (days)')
        plt.ylabel('Energy')
        #plt.ylim(8.0e20,8.15e20)
        plt.legend(loc=0)
        plt.savefig('energys.png', format='PNG')

        plt.figure(1)
        plt.plot(days, (pot_enstrophy - pot_enstrophy[0])/pot_enstrophy[0])
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Time (days)')
        plt.ylabel('Normalized changes in potential enstrophy')
        #plt.ylim(0.74, 0.78)
        plt.savefig('enstrophy.png', format='PNG')
        print(("Change in potential enstrophy = %e " % (np.abs(pot_enstrophy[-1] - pot_enstrophy[0])/pot_enstrophy[0])))

#        plt.figure(5)
#        plt.plot(days, (mass-mass[0])/mass[0])
#        plt.xlabel('Time (days)')
#        plt.ylabel('Normalized changes in mass')
#        #plt.ylim(1.175e18, 1.225e18)
#        plt.savefig('mass.png', format='PNG')
#        print(("Change in mass = %e " % (np.abs(mass[-1] - mass[0])/mass[0])))

#        plt.figure(7)
#        plt.plot(days, pv_max, '-.', label='PV max')
#        plt.plot(days, pv_min, '--', label='PV min')
#        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#        #plt.ylim([-0.0001, 0.0001])
#        plt.xlabel('Days')
#        plt.ylabel('Max/Min potential vorticity')
#        plt.legend(loc=1)
#        plt.savefig('pv_max_min.png', format='PNG')

#        plt.figure(8)
#        plt.plot(days, total_vorticity)
#        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#        #plt.ylim([-0.0001, 0.0001])
#        plt.xlabel('Days')
#        plt.ylabel('Total absolute vorticity')
#        plt.savefig('aVort_total.png', format='PNG')
#        if np.abs(total_vorticity[0]) > 1e-10: 
#            print(("Change in total vorticity = %e " % (np.abs(total_vorticity[-1] - total_vorticity[0])/total_vorticity[0])))
#        print("Initial and final total vorticity:%.15e, %.15e" % (total_vorticity[0], total_vorticity[-1]))

        if c.test_case == 2 or c.test_case == 12:
            plt.figure(2); 
            plt.plot(days, error1[:,0], '--', label=r'$L^1$ norm')
            plt.plot(days, error2[:,0], '-', label=r'$L^2$ norm')
            plt.plot(days, errorInf[:,0], '-.', label=r'$L^\infty$ norm')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.legend(loc=1)
            plt.xlabel('Time (days)')
            plt.ylabel('Relative error in thickness')
            plt.savefig('error-h.png', format='PNG')

            plt.figure(3); 
            plt.plot(days, error1[:,1], '--', label=r'$L^1$ norm')
            plt.plot(days, error2[:,1], '-', label=r'$L^2$ norm')
            plt.plot(days, errorInf[:,1], '-.', label=r'$L^\infty$ norm')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.legend(loc=1)
            plt.xlabel('Time (days)')
            plt.ylabel('Relative error in vorticity')
            plt.savefig('error-vorticity.png', format='PNG')

            plt.figure(4); 
            plt.plot(days, error1[:,2], '--', label=r'$L^1$ norm')
            plt.plot(days, error2[:,2], '-', label=r'$L^2$ norm')
            plt.plot(days, errorInf[:,2], '-.', label=r'$L^\infty$ norm')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.legend(loc=1)
            plt.xlabel('Time (days)')
            plt.ylabel('Absolute error in divergence')
            plt.savefig('error-divergence.png', format='PNG')


    if c.test_case == 2:
        for iLayer in range(c.nLayers):
            print("\nlayer %d:" % iLayer)
            print("Final l2 errors for thickness, vorticity, and divergence:")
            print(("                    %e,        %e,     %e" % (error2[-1,0,iLayer], error2[-1,1,iLayer], error2[-1,2,iLayer])))

            print("Final l8 errors for thickness, vorticity, and divergence:")
            print(("                    %e,        %e,     %e" % (errorInf[-1,0,iLayer], errorInf[-1,1,iLayer], errorInf[-1,2,iLayer])))


    print(('CPU time used: %f seconds' % (t1-t0)))
    print(('Walltime used: %f seconds' % (t1a-t0a)))

        
if __name__ == '__main__':
    main( )


            
        
    
