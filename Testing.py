import numpy as np
import time
from scipy.sparse import isspmatrix_bsr, isspmatrix_csr
from scipy.sparse.linalg import factorized, splu
from swe_comp import swe_comp as cmp

def run_tests(g, c, s, vc, poisson):


    if False:
        # To compare various solvers (direct, amgx, amg) for the coupled elliptic equation
        # The solvers are invoked directly
        from scipy.sparse.linalg import spsolve
        import pyamgx
        import cupy as cp
        import cupyx

        ##########################################################
        if True:
            ## Data from SWSTC #2 (stationary zonal flow over the global sphere)
            if not c.on_a_global_sphere:
                raise ValueError("Must use a global spheric domain")

            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            s.thickness[:] = gh / c.gravity
            h0 = gh0 / c.gravity

            s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            s.divergence[:] = 0.

            psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
            psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
            psi_cell_true -= psi_cell_true[0]
            phi_cell_true = np.zeros(g.nCells)

        elif False:
            # SWSTC #2, with a stationary analytic solution, modified for the northern hemisphere
            if c.on_a_global_sphere:
                print("This is a test case on the northern hemisphere.")
                raise ValueError("Must use a bounded domain")
            
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            s.thickness[:] = gh / c.gravity
            h0 = gh0 / c.gravity

            s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            s.divergence[:] = 0.
            
            psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
            psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
            phi_cell_true = np.zeros(g.nCells)

            s.SS0 = np.sum((s.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)
            
        elif False:
            ## Data from Test Case #22 (a free gyre in the northern atlantic)
            if c.on_a_global_sphere:
                print("This is a test case in the northern Atlantic")
                print("A global spheric domain cannot be used.")
                raise ValueError("Must use a bounded domain")
                
            latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
            lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

            latmid = 0.5*(latmin+latmax)
            latwidth = latmax - latmin

            lonmid = 0.5*(lonmin+lonmax)
            lonwidth = lonmax - lonmin

            pi = np.pi; sin = np.sin; exp = np.exp
            r = c.sphere_radius
        
            d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-1.1))**2/.3**2)
            f0 = np.mean(g.fCell)
            s.thickness[:] = 4000.
            psi_cell_true = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
            psi_cell_true *= c.gravity / f0 * s.thickness
            phi_cell_true = np.zeros(g.nCells)
            s.vorticity = vc.discrete_laplace_v(psi_cell_true)
            s.vorticity /= s.thickness
            s.divergence[:] = 0.

        else:
            raise ValueError("No data are chosen!")

        # To prepare the rhs
        s.vortdiv[:g.nCells] = s.vorticity * g.areaCell
        s.vortdiv[g.nCells:] = s.divergence * g.areaCell
        if c.on_a_global_sphere:
            # A global domain with no boundary
            s.vortdiv[0] = 0.   # Set first element to zeor to make psi_cell[0] zero
        else:
            # A bounded domain with homogeneous Dirichlet for the psi and
            # homogeneous Neumann for phi
            s.vortdiv[vc.cellBoundary-1] = 0.   # Set boundary elements to zeor to make psi_cell zero there
            
        s.vortdiv[g.nCells] = 0.   # Set first element to zeor to make phi_cell[0] zero
        
        wall0 = time.time( )
        s.thickness_edge = vc.cell2edge(s.thickness)
        poisson.update(s.thickness_edge, vc, c, g)
        wall1 = time.time( )
        print(("Wall time for updating matrices: %f" % (wall1-wall0,)))

        ########################################################################
        if False:
            print("")
            print("Solve the linear system by the direct method") # Obselete, may be removed in future
            wall0 = time.time( )
            x = spsolve(vc.coefM, s.vortdiv)
            wall1 = time.time( )

            print("")
            print(("Wall time for direct method: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = x[:g.nCells]
            s.phi_cell[:] = x[g.nCells:]
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)


        ###########################################################################
        if False:
            print("")
            print("Solve the coupled linear system by AMGX")  # Obselete

            pyamgx.initialize( )

            # Initialize config, resources and mode:
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json')
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json')
            cfg = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": 1, 
                        "algorithm": "AGGREGATION", 
                        "print_vis_data": 0, 
                        "solver": "AMG", 
                        "smoother": {
                            "relaxation_factor": 0.8, 
                            "scope": "jacobi", 
                            "solver": "BLOCK_JACOBI", 
                            "monitor_residual": 0, 
                            "print_solve_stats": 0
                        }, 
                        "print_solve_stats": 0, 
                        "presweeps": 2, 
                        "selector": "SIZE_2", 
                        "coarse_solver": "NOSOLVER", 
                        "max_iters": 2, 
                        "monitor_residual": 0, 
                        "store_res_history": 0, 
                        "scope": "amg_solver", 
                        "max_levels": 100, 
                        "postsweeps": 2, 
                        "cycle": "V"
                    }, 
                    "solver": "PCGF", 
                    "print_solve_stats": 1, 
                    "obtain_timings": 1, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "RELATIVE_INI_CORE", 
                    "scope": "main", 
                    "tolerance": c.err_tol,
                    "norm": "L2"
                }
            })

            
            rsc = pyamgx.Resources().create_simple(cfg)
            mode = 'dDDI'

            # Create solver:
            slv = pyamgx.Solver().create(rsc, cfg, mode)

            # Create matrices and vectors:
            d_A = pyamgx.Matrix().create(rsc, mode)
            d_x = pyamgx.Vector().create(rsc, mode)
            d_b = pyamgx.Vector().create(rsc, mode)

            #vc.update_matrix_for_coupled_elliptic(
            #d_A.upload_CSR(vc.coefM)
            d_A.upload(vc.coefM.indptr, vc.coefM.indices, vc.coefM.data)

            d_vortdiv = cp.array(s.vortdiv)
            d_b.upload_raw(d_vortdiv.data, d_vortdiv.size)
            x = np.zeros(g.nCells*2)
            d_x.upload(x)

            # Setup and solve system:
            cpu0 = time.clock( )
            wall0 = time.time( )
            slv.setup(d_A)
            slv.solve(d_b, d_x)
            cpu1 = time.clock( )
            wall1 = time.time( )

            d_x.download(x)

            # Clean up:
            d_A.destroy()
            d_x.destroy()
            d_b.destroy()
            slv.destroy()
            rsc.destroy()
            cfg.destroy()
            pyamgx.finalize()

            print("")
            print(("CPU time for AMGX: %f" % (cpu1-cpu0,)))
            print(("Wall time for AMGX: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = x[:g.nCells]
            s.phi_cell[:] = x[g.nCells:]
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)        


        ###########################################################################
        if False:
            print("")
            print("Solve the linear system by pyamg")   # Obsolete
            from pyamg import rootnode_solver
            A_spd = -vc.coefM
            B = np.ones((A_spd.shape[0],1), dtype=A_spd.dtype); BH = B.copy()

            cpu0 = time.clock( )
            wall0 = time.time( )
            amg_solver = rootnode_solver(A_spd, B=B, BH=BH,
                strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), \
                                    None, None, None, None, None, None, None, None, None, None, \
                                    None, None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")

            res = []
            x0 = np.zeros(g.nCells*2)
            x = amg_solver.solve(s.vortdiv, x0=x0, tol=c.err_tol, residuals=res)
            cpu1 = time.clock( )
            wall1 = time.time( )

            x *= -1

            print("")
            print(amg_solver)
            print(res)
            print("AMG, nIter = %d" % (len(res),))
            print(("CPU time pyAMG: %f" % (cpu1-cpu0,)))
            print(("Wall time pyAMG: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = x[:g.nCells]
            s.phi_cell[:] = x[g.nCells:]
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)        

        ###########################################################################
        if False:
            print("")
            print("Solve the coupled linear system with an iterative scheme and pyamg")

            from pyamg import rootnode_solver
            A11 = -poisson.A11
            A12 = -poisson.A12
            A21 = -poisson.A21
            A22 = -poisson.A22

            B11 = np.ones((A11.shape[0],1), dtype=A11.dtype); BH11 = B11.copy()
            A11_solver = rootnode_solver(A11, B=B11, BH=BH11,
                strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), \
                                    None, None, None, None, None, None, None, None, None, None, \
                                    None, None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")

            B22 = np.ones((A22.shape[0],1), dtype=A22.dtype); BH22 = B22.copy()
            A22_solver = rootnode_solver(A22, B=B22, BH=BH22,
                strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), \
                                    None, None, None, None, None, None, None, None, None, None, \
                                    None, None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")

            print("")
            print(A11_solver)
            print(A22_solver)
            wall0 = time.time( )
            x_res = []
            y_res = []
            x0 = np.zeros(g.nCells); x = x0.copy( )
            y0 = np.zeros(g.nCells); y = y0.copy( )
            for k in np.arange(10):
                xtemp = x0; ytemp = y0
                x0 = x; y0 = y
                x = xtemp; y = ytemp
                b1 = s.vortdiv[:g.nCells] - A12.dot(y0)
                b2 = s.vortdiv[g.nCells:] - A21.dot(x0)
                x = A11_solver.solve(b1, x0=x0, tol=c.err_tol, residuals=x_res)
                y = A22_solver.solve(b2, x0=y0, tol=c.err_tol, residuals=y_res)
                print("k = %d,  AMG nIters = %d, %d" % (k, len(x_res), len(y_res)))
                print(x_res)
                print(y_res)

            wall1 = time.time( )

            print(("Wall time by iterative pyAMG: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = -x[:]
            s.phi_cell[:] = -y[:]
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)

        ###########################################################################
        if False:
            print("")
            print("Solve the coupled linear system with an iterative scheme and amgx")
            pyamgx.initialize()

            # Initialize config, resources and mode:
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json')
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json')
            err_tol = 1e-8*1e-5*np.mean(g.areaCell)*np.sqrt(g.nCells)  # For vorticity
            cfg1 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": 1, 
                        "algorithm": "AGGREGATION", 
                        "print_vis_data": 0, 
                        "solver": "AMG", 
                        "smoother": {
                            "relaxation_factor": 0.8, 
                            "scope": "jacobi", 
                            "solver": "BLOCK_JACOBI", 
                            "monitor_residual": 0, 
                            "print_solve_stats": 0
                        }, 
                        "print_solve_stats": 0, 
                        "presweeps": 2, 
                        "selector": "SIZE_2", 
                        "coarse_solver": "NOSOLVER", 
                        "max_iters": 2, 
                        "monitor_residual": 0, 
                        "store_res_history": 0, 
                        "scope": "amg_solver", 
                        "max_levels": 100, 
                        "postsweeps": 2, 
                        "cycle": "V"
                    }, 
                    "solver": "PCGF", 
                    "print_solve_stats": 0, 
                    "obtain_timings": 0, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            err_tol = 1e-8*1e-6*np.mean(g.areaCell)*np.sqrt(g.nCells) # For divergence
            cfg2 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": 1, 
                        "algorithm": "AGGREGATION", 
                        "print_vis_data": 0, 
                        "solver": "AMG", 
                        "smoother": {
                            "relaxation_factor": 0.8, 
                            "scope": "jacobi", 
                            "solver": "BLOCK_JACOBI", 
                            "monitor_residual": 0, 
                            "print_solve_stats": 0
                        }, 
                        "print_solve_stats": 0, 
                        "presweeps": 2, 
                        "selector": "SIZE_2", 
                        "coarse_solver": "NOSOLVER", 
                        "max_iters": 2, 
                        "monitor_residual": 0, 
                        "store_res_history": 0, 
                        "scope": "amg_solver", 
                        "max_levels": 100, 
                        "postsweeps": 2, 
                        "cycle": "V"
                    }, 
                    "solver": "PCGF", 
                    "print_solve_stats": 0, 
                    "obtain_timings": 0, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            rsc1 = pyamgx.Resources().create_simple(cfg1)
            rsc2 = pyamgx.Resources().create_simple(cfg2)
            mode = 'dDDI'

            # Create solver:
            slv11 = pyamgx.Solver().create(rsc1, cfg1, mode)
            slv22 = pyamgx.Solver().create(rsc2, cfg2, mode)

            # Create matrices and vectors:
            d_A11 = pyamgx.Matrix().create(rsc1, mode)
            d_x = pyamgx.Vector().create(rsc1, mode)
            d_b1 = pyamgx.Vector().create(rsc1, mode)
            d_A22 = pyamgx.Matrix().create(rsc2, mode)
            d_y = pyamgx.Vector().create(rsc2, mode)
            d_b2 = pyamgx.Vector().create(rsc2, mode)

            d_A11.upload_CSR(poisson.A11)
            d_A22.upload_CSR(poisson.A22)

            b1 = s.vortdiv[:g.nCells]
            d_b1.upload(b1)
            x = np.zeros(g.nCells)
            d_x.upload(x)
            b2 = s.vortdiv[g.nCells:]
            d_b2.upload(b2)
            y = np.zeros(g.nCells)
            d_y.upload(y)

            # Setup and solve system:
            slv11.setup(d_A11)
            slv22.setup(d_A22)

            wall0 = time.time( )
            for k in np.arange(10):
                b1 = s.vortdiv[:g.nCells] - poisson.A12.dot(y)
                b2 = s.vortdiv[g.nCells:] - poisson.A21.dot(x)
                d_b1.upload(b1)
                d_b2.upload(b2)
                slv11.solve(d_b1, d_x)
                slv22.solve(d_b2, d_y)
                d_x.download(x)
                d_y.download(y)

            wall1 = time.time( )

            # Clean up:
            d_A11.destroy()
            d_A22.destroy()
            d_x.destroy()
            d_y.destroy()
            d_b1.destroy()
            d_b2.destroy()
            slv11.destroy()
            slv22.destroy()
            rsc1.destroy()
            cfg1.destroy()
            rsc2.destroy()
            cfg2.destroy()
            pyamgx.finalize()

            print("")
            print(("Wall time for AMGX: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = -x[:]
            s.phi_cell[:] = -y[:]
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)
#            print("psi_cell_true = ")
#            print(psi_cell_true)
#            print("s.psi_cell = ")
#            print(s.psi_cell)
        
        ###########################################################################
        if True:
            print("")
            print("Solve the coupled linear system with an iterative scheme, amgx, upload_raw and download_raw")
            pyamgx.initialize()

            # Initialize config, resources and mode:
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json')
            #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json')
            err_tol = 1e-8*1e-5*np.mean(g.areaCell)*np.sqrt(g.nCells)  # For vorticity
            cfg1 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": 1, 
                        "algorithm": "AGGREGATION", 
                        "print_vis_data": 0, 
                        "solver": "AMG", 
                        "smoother": {
                            "relaxation_factor": 0.8, 
                            "scope": "jacobi", 
                            "solver": "BLOCK_JACOBI", 
                            "monitor_residual": 0, 
                            "print_solve_stats": 0
                        }, 
                        "print_solve_stats": 0, 
                        "presweeps": 2, 
                        "selector": "SIZE_2", 
                        "coarse_solver": "NOSOLVER", 
                        "max_iters": 2, 
                        "monitor_residual": 0, 
                        "store_res_history": 0, 
                        "scope": "amg_solver", 
                        "max_levels": 100, 
                        "postsweeps": 2, 
                        "cycle": "V"
                    }, 
                    "solver": "PCGF", 
                    "print_solve_stats": 0, 
                    "obtain_timings": 0, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            err_tol = 1e-8*1e-6*np.mean(g.areaCell)*np.sqrt(g.nCells) # For divergence
            cfg2 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": 1, 
                        "algorithm": "AGGREGATION", 
                        "print_vis_data": 0, 
                        "solver": "AMG", 
                        "smoother": {
                            "relaxation_factor": 0.8, 
                            "scope": "jacobi", 
                            "solver": "BLOCK_JACOBI", 
                            "monitor_residual": 0, 
                            "print_solve_stats": 0
                        }, 
                        "print_solve_stats": 0, 
                        "presweeps": 2, 
                        "selector": "SIZE_2", 
                        "coarse_solver": "NOSOLVER", 
                        "max_iters": 2, 
                        "monitor_residual": 0, 
                        "store_res_history": 0, 
                        "scope": "amg_solver", 
                        "max_levels": 100, 
                        "postsweeps": 2, 
                        "cycle": "V"
                    }, 
                    "solver": "PCGF", 
                    "print_solve_stats": 0, 
                    "obtain_timings": 0, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            rsc1 = pyamgx.Resources().create_simple(cfg1)
            rsc2 = pyamgx.Resources().create_simple(cfg2)
            mode = 'dDDI'

            # Create solver:
            slv11 = pyamgx.Solver().create(rsc1, cfg1, mode)
            slv22 = pyamgx.Solver().create(rsc2, cfg2, mode)

            # Create matrices and vectors:
            d_A11 = pyamgx.Matrix().create(rsc1, mode)
            d_x = pyamgx.Vector().create(rsc1, mode)
            x_cp = cp.zeros(g.nCells); x = np.zeros(g.nCells)
            d_b1 = pyamgx.Vector().create(rsc1, mode); b1_cp = cp.zeros(g.nCells)
            d_A22 = pyamgx.Matrix().create(rsc2, mode)
            d_y = pyamgx.Vector().create(rsc2, mode)
            y_cp = cp.zeros(g.nCells); y = np.zeros(g.nCells)
            d_b2 = pyamgx.Vector().create(rsc2, mode); b2_cp = cp.zeros(g.nCells)

            # Export A11 out for more testing
#            from scipy.io import mmwrite
#            mmwrite('A11-40962', poisson.A11)
            
            t0 = time.time( )
            A11_cp = cupyx.scipy.sparse.csr_matrix(poisson.A11)
            t1 = time.time( )
            print("Time to convert scipy CSR into cupy CSR: %f" % (t1-t0))
            
            d_A11.upload_CSR(poisson.A11)
            t2 = time.time( )
            print("Time to upload scipy CSR to AMGX solver: %f" % (t2-t1))
            
            d_A11.upload_CSR(A11_cp)
            t3 = time.time( )
            print("Time to upload cupy CSR to AMGX solver: %f" % (t3-t2))
            
            d_A11.upload_CSR(A11_cp)
            t4 = time.time( )
            print("Time to upload cupy CSR to AMGX solver: %f" % (t4-t3))
            
            d_A11.upload_CSR(A11_cp)
            t5 = time.time( )
            print("Time to upload cupy CSR to AMGX solver: %f" % (t5-t4))

            
            A22_cp = cupyx.scipy.sparse.csr_matrix(poisson.A22)
            d_A22.upload_CSR(poisson.A22)
#            d_A22.upload_CSR(A22_cp)
            
            A12_cp = cupyx.scipy.sparse.csr_matrix(poisson.A12)
            A21_cp = cupyx.scipy.sparse.csr_matrix(poisson.A21)

            

#            raise ValueError( )
        
            vortdiv_cp = cp.asarray(s.vortdiv)
            b1_cp = cp.asarray(s.vortdiv[:g.nCells])
            d_b1.upload_raw(b1_cp.data, b1_cp.size)
            d_x.upload_raw(x_cp.data, x_cp.size)
            b2_cp = cp.asarray(s.vortdiv[g.nCells:])
            d_b2.upload_raw(b2_cp.data, b2_cp.size)
            d_y.upload_raw(y_cp.data, y_cp.size)

            # Setup and solve system:
            slv11.setup(d_A11)
            slv22.setup(d_A22)

            wall0 = time.time( )
            for k in np.arange(10):
                b1_cp = vortdiv_cp[:g.nCells] - A12_cp.dot(y_cp)
                b2_cp = vortdiv_cp[g.nCells:] - A21_cp.dot(x_cp)
#                print("vortdiv_cp[g.nCells:] = ")
#                print(vortdiv_cp[g.nCells:])
#                print("A21_cp.dot(x_cp) = ")
#                print(A21_cp.dot(x_cp))
#                print("b1_cp")
#                print(b1_cp)
#                print("b2_cp")
#                print(b2_cp)
#                d_b1.upload_raw(b1_cp.data, b1_cp.size)
#                d_b2.upload_raw(b2_cp.data, b2_cp.size)
                d_b1.upload(b1_cp)
                d_b2.upload(b2_cp)
                slv11.solve(d_b1, d_x)
                slv22.solve(d_b2, d_y)
                d_x.download_raw(x_cp.data)
                d_y.download_raw(y_cp.data)
#                d_x.download(x_cp)     #This does not work yet. pyamgx does not 
#                d_y.download(y_cp)     #download to cupy arrays yet.
#                d_x.download(x)
#                d_y.download(y)
#                x_cp = cp.asarray(x)
#                y_cp = cp.asarray(y)
#                print("x_cp = ")
#                print(x_cp)
#                print("y_cp = ")
#                print(y_cp)

            wall1 = time.time( )

            # Clean up:
            d_A11.destroy()
            d_A22.destroy()
            d_x.destroy()
            d_y.destroy()
            d_b1.destroy()
            d_b2.destroy()
            slv11.destroy()
            slv22.destroy()
            rsc1.destroy()
            cfg1.destroy()
            rsc2.destroy()
            cfg2.destroy()
            pyamgx.finalize()

            print("")
            print(("Wall time for AMGX: %f" % (wall1-wall0,)))

            # Compute the errors
            s.psi_cell[:] = -x_cp.get( )    # Not sure why this needs to be negated
            s.phi_cell[:] = -y_cp.get( )    # But it works for the moment
            l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
            l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
            l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
            l2 = np.sqrt(l2)
            print("Errors in psi")
            print("L infinity error = ", l8)
            print("L^2 error        = ", l2)

            #raise ValueError("Debugging")


    if False:
        print("")
        print("Testing the EllipticCpl2 object for the coupled elliptic equation")

        ##########################################################
        if True:
            ## Data from SWSTC #2 (stationary zonal flow over the global sphere)
            if not c.on_a_global_sphere:
                raise ValueError("Must use a global spheric domain")

            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            s.thickness[:] = gh / c.gravity
            h0 = gh0 / c.gravity

            s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            s.divergence[:] = 0.

            psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
            psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
            psi_cell_true -= psi_cell_true[0]
            phi_cell_true = np.zeros(g.nCells)

        elif False:
            # SWSTC #2, with a stationary analytic solution, modified for the northern hemisphere
            if c.on_a_global_sphere:
                print("This is a test case on the northern hemisphere.")
                raise ValueError("Must use a bounded domain")
            
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            s.thickness[:] = gh / c.gravity
            h0 = gh0 / c.gravity

            s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            s.divergence[:] = 0.
            
            psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
            psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
            phi_cell_true = np.zeros(g.nCells)

            s.SS0 = np.sum((s.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)
            
        elif False:
            ## Data from Test Case #22 (a free gyre in the northern atlantic)
            if c.on_a_global_sphere:
                print("This is a test case in the northern Atlantic")
                print("A global spheric domain cannot be used.")
                raise ValueError("Must use a bounded domain")
                
            latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
            lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

            latmid = 0.5*(latmin+latmax)
            latwidth = latmax - latmin

            lonmid = 0.5*(lonmin+lonmax)
            lonwidth = lonmax - lonmin

            pi = np.pi; sin = np.sin; exp = np.exp
            r = c.sphere_radius
        
            d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-1.1))**2/.3**2)
            f0 = np.mean(g.fCell)
            s.thickness[:] = 4000.
            psi_cell_true = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
            psi_cell_true *= c.gravity / f0 * s.thickness
            phi_cell_true = np.zeros(g.nCells)
            s.vorticity = vc.discrete_laplace_v(psi_cell_true)
            s.vorticity /= s.thickness
            s.divergence[:] = 0.

        # To prepare the rhs
        s.vortdiv[:g.nCells] = s.vorticity * g.areaCell
        s.vortdiv[g.nCells:] = s.divergence * g.areaCell
        if c.on_a_global_sphere:
            # A global domain with no boundary
            s.vortdiv[0] = 0.   # Set first element to zeor to make psi_cell[0] zero
        else:
            # A bounded domain with homogeneous Dirichlet for the psi and
            # homogeneous Neumann for phi
            s.vortdiv[vc.cellBoundary-1] = 0.   # Set boundary elements to zeor to make psi_cell zero there
            
        s.vortdiv[g.nCells] = 0.   # Set first element to zeor to make phi_cell[0] zero

        
        cpu0 = time.clock( )
        wall0 = time.time( )
        s.thickness_edge = vc.cell2edge(s.thickness)
        poisson.update(s.thickness_edge, vc, c, g)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for updating matrices: %f" % (cpu1-cpu0,)))
        print(("Wall time for updating matrices: %f" % (wall1-wall0,)))

        cpu0 = time.clock( )
        wall0 = time.time( )
        x = np.zeros(g.nCells); y = np.zeros(g.nCells)
        b1 = s.vortdiv[:g.nCells]; b2 = s.vortdiv[g.nCells:]
        poisson.solve(b1, b2, x, y)
        cpu1 = time.clock( )
        wall1 = time.time( )

        print("Solver used: %s" % c.linear_solver)
        print(("CPU time: %f" % (cpu1-cpu0,)))
        print(("Wall time: %f" % (wall1-wall0,)))

        # Compute the errors
        s.psi_cell[:] = x[:]
        s.phi_cell[:] = y[:]
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)

    if False:
        # Test multiplation of CuPy sparse matrices, and the matrix upload method
        # for pyAMGX
        import pyamgx
        import cupy as cp
        import cupyx

        pyamgx.initialize()

        # Initialize config, resources and mode:
        #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json')
        #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json')
        cfg = pyamgx.Config( ).create_from_dict({    
            "config_version": 2, 
            "determinism_flag": 0, 
            "solver": {
                "preconditioner": {
                    "print_grid_stats": 1, 
                    "algorithm": "AGGREGATION", 
                    "print_vis_data": 0, 
                    "solver": "AMG", 
                    "smoother": {
                        "relaxation_factor": 0.8, 
                        "scope": "jacobi", 
                        "solver": "BLOCK_JACOBI", 
                        "monitor_residual": 0, 
                        "print_solve_stats": 0
                    }, 
                    "print_solve_stats": 0, 
                    "presweeps": 2, 
                    "selector": "SIZE_2", 
                    "coarse_solver": "NOSOLVER", 
                    "max_iters": 2, 
                    "monitor_residual": 0, 
                    "store_res_history": 0, 
                    "scope": "amg_solver", 
                    "max_levels": 100, 
                    "postsweeps": 2, 
                    "cycle": "V"
                }, 
                "solver": "PCGF", 
                "print_solve_stats": 1, 
                "obtain_timings": 1, 
                "max_iters": c.max_iters, 
                "monitor_residual": 1, 
                "convergence": "ABSOLUTE", 
                "scope": "main", 
                "tolerance": 1e-8,
                "norm": "L2"
            }
        })

        rsc = pyamgx.Resources().create_simple(cfg)
        mode = 'dDDI'

        # Create matrices and vectors:
        d_A11 = pyamgx.Matrix().create(rsc, mode)

        AC = poisson.AC
        d_AC = cupyx.scipy.sparse.csr_matrix(AC)
        mSkewgrad_td = vc.mSkewgrad_td
        d_mSkewgrad_td = cupyx.scipy.sparse.csr_matrix(mSkewgrad_td)

        thicknessInv = np.random.rand(g.nEdges)
        t0 = time.time( )
        A11 = AC.multiply(thicknessInv)
        A11 *= mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on CPU: %f" % (t1-t0))
        print("Upload matrix from host to AMGX on GPU: %f" % (t2-t1))

        thicknessInv = np.random.rand(g.nEdges)
        t0 = time.time( )
        A11 = AC.multiply(thicknessInv)
        A11 *= mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on CPU: %f" % (t1-t0))
        print("Upload matrix from host to AMGX on GPU: %f" % (t2-t1))

        thicknessInv = np.random.rand(g.nEdges)
        t0 = time.time( )
        A11 = AC.multiply(thicknessInv)
        A11 *= mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on CPU: %f" % (t1-t0))
        print("Upload matrix from host to AMGX on GPU: %f" % (t2-t1))
        
        thicknessInv = np.random.rand(g.nEdges)
        #thicknessInv_cp = cupyx.scipy.sparse.diags(thicknessInv, format='csr')
        thicknessInv_cp = cp.array(thicknessInv)
        #A11_cp = d_AC * thicknessInv_cp
        t0 = time.time( )
        A11_cp = d_AC.multiply(thicknessInv_cp)
        A11_cp *= d_mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11_cp)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on GPU, with multiply: %f" % (t1-t0))
        print("Upload matrix from device to AMGX on GPU: %f" % (t2-t1))

        thicknessInv = np.random.rand(g.nEdges)
        thicknessInv_cp = cupyx.scipy.sparse.diags(thicknessInv, format='csr')
        t0 = time.time( )
        A11_cp = d_AC * thicknessInv_cp
        A11_cp *= d_mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11_cp)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on GPU, with diagonals: %f" % (t1-t0))
        print("Upload matrix from device to AMGX on GPU: %f" % (t2-t1))

        thicknessInv = np.random.rand(g.nEdges)
        #thicknessInv_cp = cupyx.scipy.sparse.diags(thicknessInv, format='csr')
        thicknessInv_cp = cp.array(thicknessInv)
        #A11_cp = d_AC * thicknessInv_cp
        t0 = time.time( )
        A11_cp = d_AC.multiply(thicknessInv_cp)
        A11_cp *= d_mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11_cp)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on GPU, with multiply: %f" % (t1-t0))
        print("Upload matrix from device to AMGX on GPU: %f" % (t2-t1))
        
        thicknessInv = np.random.rand(g.nEdges)
        thicknessInv_cp = cupyx.scipy.sparse.diags(thicknessInv, format='csr')
        t0 = time.time( )
        A11_cp = d_AC * thicknessInv_cp
        A11_cp *= d_mSkewgrad_td
        t1 = time.time()
        d_A11.upload_CSR(A11_cp)
        t2 = time.time( )
        print("")
        print("Matrix multiplication on GPU, with diagonals: %f" % (t1-t0))
        print("Upload matrix from device to AMGX on GPU: %f" % (t2-t1))
        

    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # Both normal and tangential components are used to define the Hamiltonian
        # using the EllipticCPL object
        # vorticity and divergence derived from psi and phi

        s.thickness[:] = 4000. + np.random.rand(g.nCells) * 100
        s.thickness_edge = vc.cell2edge(s.thickness)

        psi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        if c.on_a_global_sphere:
            psi_cell_true[0] = 0.
        else:
            psi_cell_true[vc.cellBoundary[:]-1] = 0.
            
        phi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        phi_cell_true[0] = 0.

        psi_vertex = vc.cell2vertex(psi_cell_true)
        phi_vertex = vc.cell2vertex(phi_cell_true)

        hv = vc.discrete_grad_n(psi_cell_true)
        hv += vc.discrete_grad_tn(phi_vertex)
        v = hv / s.thickness_edge

        hu = vc.discrete_grad_n(phi_cell_true)
        hu -= vc.discrete_grad_td(psi_vertex)
        u = hu / s.thickness_edge
        
        s.vorticity = 0.5*vc.discrete_curl_v(v)
        s.vorticity += 0.5 * vc.vertex2cell(vc.discrete_curl_t(u))
        
        s.divergence = 0.5*vc.vertex2cell(vc.discrete_div_t(v))
        s.divergence += 0.5*vc.discrete_div_v(u)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

        l8 = np.max(np.abs(phi_cell_true[:] - s.phi_cell[:])) / np.max(np.abs(phi_cell_true[:]))
        l2 = np.sum(np.abs(phi_cell_true[:] - s.phi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(phi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in phi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # using the EllipticCPL object
        # vorticity and divergence come from SWSTC #2

        a = c.sphere_radius
        u0 = 2*np.pi*a / (12*86400)
        gh0 = 2.94e4
        gh = np.sin(g.latCell[:])**2
        gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
        s.thickness[:] = gh / c.gravity
        h0 = gh0 / c.gravity

        s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
        s.divergence[:] = 0.

        psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
        psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
        psi_cell_true -= psi_cell_true[0]
        phi_cell_true = np.zeros(g.nCells)
        
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = %e" % l8)
        print("L^2 error        = %e" % l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on a bounded domain
        # using the EllipticCPL object
        # vorticity and divergence come from TC #22

        latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
        lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

        latmid = 0.5*(latmin+latmax)
        latwidth = latmax - latmin

        lonmid = 0.5*(lonmin+lonmax)
        lonwidth = lonmax - lonmin

        pi = np.pi; sin = np.sin; exp = np.exp
        r = c.sphere_radius
        
        d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-1.1))**2/.3**2)
        f0 = np.mean(g.fCell)
        s.thickness[:] = 4000.
        psi_cell_true = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
        psi_cell_true *= c.gravity / f0 * s.thickness
        phi_cell_true = np.zeros(g.nCells)
        s.vorticity = vc.discrete_laplace_v(psi_cell_true)
        s.vorticity /= s.thickness
        s.divergence[:] = 0.

        s.psi_cell[:] = 0.
        s.phi_cell[:] = 0.
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = %e" % l8)
        print("L^2 error        = %e" % l2)        
        

    if False:
        # To run solver_diagnostics for the AMG
        print("To run solver_diagnostics for the AMG")
        
        solver_diagnostics(vc.POpn.A_spd, fname='p15km', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)

        solver_diagnostics(vc.POdn.A_spd, fname='d15km', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)
        
        

    if False:
        print("To study and compare initializaiton schemes ")
        
        sol_cell = np.cos(g.latCell)*np.sin(g.lonCell)
        sol_cell[:] -= sol_cell[0]
        sol_vertex = np.cos(g.latVertex)*np.sin(g.lonVertex)
        sol_vertex[:] -= sol_vertex[0]

        vort_cell = cmp.discrete_laplace( \
                 g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                                          sol_cell)
        vort_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, vort_cell)
        
        x_cell = np.zeros(g.nCells)
        b_cell = vort_cell[:] * g.areaCell[:]
        b_cell[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.D2s, b_cell, x_cell, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell)))))
        print(("CPU time for cg solver on primary mesh: %f" % (t1-t0,)))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = vort_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,)))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print(("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,)))
        

    if False:
        print("Repeat previous test, but using data from SWSTC #5 ")

        s.initialization(g, c)
        sol_cell = s.psi_cell[:]
        sol_cell[:] -= sol_cell[0]
        sol_vertex = s.psi_vertex[:]
        sol_vertex[:] -= sol_vertex[0]

        x_cell = np.zeros(g.nCells)
        b_cell = s.vorticity[:] * g.areaCell[:]
        b_cell[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.D2s, b_cell, x_cell, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell)))))
        print(("CPU time for cg solver on primary mesh: %f" % (t1-t0,)))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = s.vorticity_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,)))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print(("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, max_iter=c.max_iter, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,)))
        
        
    if False:
        # Test LinearAlgebra.cg solver by a small simple example
        A = csr_matrix([[2,1,0],[1,3,1],[0,1,4]])
        sol = np.array([1,0,-1])
        b = np.array([2,0,-4])

        x0 = np.array([0.,0.,0.])
        x, info = cg_scipy(A, b, x0=x0, maxiter = 200)
        print(("cg_scipy: info = %d" % info))
        print(x)
        info, nIter = cg(A, b, x0, max_iter = 100)
        print(("nIter = %d" % nIter))
        print("x = ")
        print(x0)


    if False:
        print("To test cg with incomplete cholesky as preconditioner")

        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = vc.D2s.dot(sol)
        
        A = vc.D2s.tocsr( )
        A.eliminate_zeros( )
        A_t = deepcopy(A)
        A_t.data = np.where(A_t.nonzero()[0] >= A_t.nonzero()[1], A_t.data, 0.)
        A_t.eliminate_zeros( )
        R = A_t.copy( )
        R = -R

        
        cuSparse = cuda.sparse.Sparse()
        D2s_descr = cuSparse.matdescr(matrixtype='S', fillmode='L')
        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
                                       descr=D2s_descr, csrVal=R.data, \
                                       csrRowPtr=R.indptr, csrColInd=R.indices)
        cuSparse.csric0(trans='N', m=R.shape[0], \
                        descr=D2s_descr, csrValM=R.data, csrRowPtrA=R.indptr,\
                        csrColIndA=R.indices, info=info)
#        cuSparse.csrilu0(trans='N', m=R.shape[0], \
#                        descr=D2s_descr, csrValM=R.data, csrRowPtrA=R.indptr,\
#                        csrColIndA=R.indices, info=info)
        
        # Test triangular solver 
        b1 = R.dot(sol)
        Rsolve = factorized(R)
#        lu_R = splu(R, permc_spec='NATURAL')
        t0 = time.clock( )
        #x = spsolve_triangular(R, b1, lower=True, overwrite_b=False, overwrite_A=False)
        x = Rsolve(b1)
        #x = lu_R.solve(b1)
        t1 = time.clock( )
        print(("rel error for triangular solver = %e" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for a direct solver for triangular solver: %f" % (t1-t0,)))


        Rdata = numba.cuda.to_device(R.data)
        Rptr = numba.cuda.to_device(R.indptr)
        Rind = numba.cuda.to_device(R.indices)
        R_descr = cuSparse.matdescr(matrixtype='T', fillmode='L')
#        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
#                                       descr=R_descr, csrVal=R.data, \
#                                       csrRowPtr=R.indptr, csrColInd=R.indices)
        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
                                       descr=R_descr, csrVal=Rdata, \
                                       csrRowPtr=Rptr, csrColInd=Rind)        
        b1 = R.dot(sol)
        x = np.zeros(np.size(b1))
        t0 = time.clock( )
#        cuSparse.csrsv_solve(trans='N', m=R.shape[0], alpha=1.0, \
#                             descr=R_descr, csrVal=R.data, \
#                             csrRowPtr=R.indptr, csrColInd=R.indices, info=info, x=b1, y=x)
        cuSparse.csrsv_solve(trans='N', m=R.shape[0], alpha=1.0, \
                             descr=R_descr, csrVal=Rdata, \
                             csrRowPtr=Rptr, csrColInd=Rind, info=info, x=b1, y=x)
        t1 = time.clock( )
        print(("rel error for triangular solver = %e" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for a cuda solver for triangular solver: %f" % (t1-t0,)))

        
        D2s_solve = factorized(vc.D2s)
        x1 = np.zeros(g.nCells)
        t0 = time.clock( )
        #x1[:] = vc.lu_D2s.solve(b)
        x1[:] = D2s_solve(b)
        t1 = time.clock( )
        print(("rel error = %e" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))
        
        x4 = np.zeros(g.nCells)
        t0 = time.clock( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %e" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))

        
        raise ValueError("Stop for checking.")



    if False:
        print("Compare scipy sparse dot and cupy sparse dot, using the mLaplace sparse matrix")
        
        x = np.random.rand(g.nCells)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = vc.mLaplace_v.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for mLaplace: %f" % (t1a-t0a,)))
        print(("Wall time for mLaplace: %f" % (t1b-t0b,)))

        # Create arrays on host, and transfer them to device
        y1 = np.zeros(g.nCells)

        import cupy as cp
        import cupyx
        d_mLaplace = cupyx.scipy.sparse.csr_matrix(vc.mLaplace_v)
        
        t0a = time.clock( )
        t0b = time.time( )
        d_x = cp.array(x)
        d_y1 = cp.array(y1)
        t1a = time.clock( )
        t1b = time.time( )
        
        d_y1 = d_mLaplace.dot(d_x)
        t2a = time.clock( )
        t2b = time.time( )
        
        y1 = d_y1.get( )
        t3a = time.clock( )
        t3b = time.time( )
        
        print(("Wall time for copy to device: %f" % (t1b-t0b,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))
        print(("Wall time for copy to host: %f" % (t3b-t2b,)))
        print(("Wall time for GPU computation: %f" % (t3b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))


    if False:
        # Compare cell2edge and edge2cell with their Fortran versions
        
        x = np.random.rand(g.nCells)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.cell2edge(g.cellsOnEdge, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for Fortran cell2edge: %f" % (t1a-t0a,)))
        print(("Wall time for Fortran cell2edge: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.cell2edge(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for matrix cell2edge: %f" % (t1a-t0a,)))
        print(("Wall time for matrix cell2edge: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

        x = np.random.rand(g.nEdges)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for Fortran edge2cell: %f" % (t1a-t0a,)))
        print(("Wall time for Fortran edge2cell: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.edge2cell(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for matrix edge2cell: %f" % (t1a-t0a,)))
        print(("Wall time for matrix edge2cell: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

    if False: #Test the accuracy of the Laplace operator
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        laplace_psi_true = -9/c.sphere_radius * np.cos(g.latCell) * np.sin(3*g.lonCell)
        laplace_psi_true += 9/a*np.cos(g.latCell)*np.sin(g.latCell)**2*np.sin(3*g.lonCell)
        laplace_psi_true -= 3/a*np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        laplace_psi = vc.discrete_laplace_v(psi)

        # Compute the errors
        l8 = np.max(np.abs(laplace_psi_true[:] - laplace_psi[:])) / np.max(np.abs(laplace_psi_true[:]))
        l2 = np.sum(np.abs(laplace_psi_true[:] - laplace_psi[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(laplace_psi_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in the discrete Laplace operator:")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

    if False: #Test the accuracy of div ( grad psi_h)
              # versus Delta(psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = -9/c.sphere_radius * np.cos(g.latCell) * np.sin(3*g.lonCell)
        result_true += 9/a*np.cos(g.latCell)*np.sin(g.latCell)**2*np.sin(3*g.lonCell)
        result_true -= 3/a*np.cos(g.latCell)**3 * np.sin(3*g.lonCell)

        result_approx = vc.discrete_div_v(vc.discrete_grad_n(psi))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in the div( q grad(psi)):")
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)

    if False: #Test the accuracy of divergence ( q times grad psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = -3*np.sin(g.latCell)*np.cos(g.latCell)*np.sin(g.lonCell)*np.cos(3*g.lonCell)
        result_true -= 9*np.sin(g.latCell)*np.cos(g.latCell)*np.sin(3*g.lonCell)*np.cos(g.lonCell)
        result_true += 9*np.cos(g.latCell)*np.sin(g.latCell)**3*np.sin(3*g.lonCell)*np.cos(g.lonCell)
        result_true -= 6*np.cos(g.latCell)**3*np.sin(g.latCell)*np.sin(3*g.lonCell)*np.cos(g.lonCell)
        result_true *= 1e-8/a

        q_e = vc.cell2edge(q)
        result_approx = vc.discrete_div_v(q_e * vc.discrete_grad_n(psi))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in the div( q grad(psi)):")
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)


    if False: #Test the accuracy of nabla times (q hat times grad psi tilde)
             # versus nabla x (q nabla psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = np.sin(g.latCell)**2*np.sin(3*g.lonCell)*np.sin(g.lonCell)
        result_true -= np.cos(g.latCell)**2*np.cos(g.lonCell)*np.cos(3*g.lonCell)
        result_true *= 3.e-8*np.cos(g.latCell)/a

        q_e = vc.cell2edge(q)
        psi_vertex = vc.cell2vertex(psi)
        result_approx = vc.discrete_curl_v(q_e * vc.discrete_grad_tn(psi_vertex))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors:")
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)

    if False: # Test the accuracy of tilde(nabla times (q hat times grad psi))
             # versus nabla x (q nabla psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = np.sin(g.latCell)**2*np.sin(3*g.lonCell)*np.sin(g.lonCell)
        result_true -= np.cos(g.latCell)**2*np.cos(g.lonCell)*np.cos(3*g.lonCell)
        result_true *= 3.e-8*np.cos(g.latCell)/a

        q_e = vc.cell2edge(q)
        result_approx = vc.vertex2cell(vc.discrete_curl_t(q_e * vc.discrete_grad_n(psi)))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in the curl ( q grad(psi tilde)):")
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)


    if False: # Test the accuracy of tilde(nabla dot (psi hat times grad perp q))
             # versus nabla x (q nabla psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = np.sin(g.latCell)**2*np.sin(3*g.lonCell)*np.sin(g.lonCell)
        result_true -= np.cos(g.latCell)**2*np.cos(g.lonCell)*np.cos(3*g.lonCell)
        result_true *= 3.e-8*np.cos(g.latCell)/a

        psi_edge = vc.cell2edge(psi)
        result_approx = vc.vertex2cell(vc.discrete_div_t(psi_edge * vc.discrete_skewgrad_t(q)))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in the curl ( q grad(psi tilde)):")
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)


    if False: # Test the accuracy of nabla dot (psi hat times grad perp q tilde)
             # versus nabla x (q nabla psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = np.sin(g.latCell)**2*np.sin(3*g.lonCell)*np.sin(g.lonCell)
        result_true -= np.cos(g.latCell)**2*np.cos(g.lonCell)*np.cos(3*g.lonCell)
        result_true *= 3.e-8*np.cos(g.latCell)/a

        psi_edge = vc.cell2edge(psi)
        q_vertex = vc.cell2vertex(q)
        result_approx = vc.discrete_div_v(psi_edge * vc.discrete_skewgrad_nd(q_vertex))

        # Compute the errors
        l8 = np.max(np.abs(result_true[:] - result_approx[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors:")
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)


    if False: # Test the accuracy of 2*tilde(nabla perp beta tilde dot nabla alpha) and the accuracy of
             # -2*tilde(nabla perp alpha tilde dot nabla beta) 
             # versus nabla x (q nabla psi)
        psi = np.cos(g.latCell)**3 * np.sin(3*g.lonCell)
        psi *= c.sphere_radius
        q = 1.e-8*np.sin(g.latCell)*np.cos(g.lonCell)
        a = c.sphere_radius

        result_true = np.sin(g.latCell)**2*np.sin(3*g.lonCell)*np.sin(g.lonCell)
        result_true -= np.cos(g.latCell)**2*np.cos(g.lonCell)*np.cos(3*g.lonCell)
        result_true *= 3.e-8*np.cos(g.latCell)/a

        psi_vertex = vc.cell2vertex(psi)
        q_vertex = vc.cell2vertex(q)
        result_approx1 = -2*vc.edge2cell(vc.discrete_skewgrad_nd(psi_vertex)* vc.discrete_grad_n(q))
        result_approx2 = 2*vc.edge2cell(vc.discrete_skewgrad_nd(q_vertex)* vc.discrete_grad_n(psi))

        # Compute the errors
        print("Error in result_approx1:")
        l8 = np.max(np.abs(result_true[:] - result_approx1[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx1[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx1)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)

        print(" ")
        print("Error in result_approx2:")
        l8 = np.max(np.abs(result_true[:] - result_approx2[:])) / np.max(np.abs(result_true[:]))
        l2 = np.sum(np.abs(result_true[:] - result_approx2[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(result_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("max of result_true = %e" % np.max(np.abs(result_true)))
        print("max of result_approx = %e" % np.max(np.abs(result_approx2)))
        print("L^2 error        = ", l2)        
        print("L infinity error = ", l8)
        raise ValueError


    if False: # Study the accuracy of the vertex2cell, cell2vertex, and edge2cell mappings 
             # 
             # 
        psi_cell_true = np.cos(g.latCell)**3 * np.sin(4*g.lonCell)
        psi_cell_true *= c.sphere_radius
        psi_vertex_true = np.cos(g.latVertex)**3 * np.sin(4*g.lonVertex)
        psi_vertex_true *= c.sphere_radius
        psi_edge_true = np.cos(g.latEdge)**3 * np.sin(4*g.lonEdge)
        psi_edge_true *= c.sphere_radius

        psi_cell1 = vc.vertex2cell(psi_vertex_true)
        psi_vertex = vc.cell2vertex(psi_cell_true)
        psi_cell2 = vc.edge2cell(psi_edge_true)
        
        # Compute the errors
        print("Error in cell2vertex:")
        l8 = np.max(np.abs(psi_vertex_true[:] - psi_vertex[:])) / np.max(np.abs(psi_vertex_true[:]))
        l2 = np.sum(np.abs(psi_vertex_true[:] - psi_vertex[:])**2 * g.areaTriangle[:])
        l2 /=  np.sum(np.abs(psi_vertex_true[:])**2 * g.areaTriangle[:])
        l2 = np.sqrt(l2)
        print("L^2 error        = %e" % l2)        
        print("L infinity error = %e" % l8)

        print("Error in vertex2cell:")
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell1[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell1[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("L^2 error        = %e" % l2)        
        print("L infinity error = %e" % l8)

        print("Error in edge2cell:")
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell2[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell2[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("L^2 error        = %e" % l2)        
        print("L infinity error = %e" % l8)


    if True: # misc GPU tests
        print(type(g.latCell))
        print(type(g.latCell.get()))
    
