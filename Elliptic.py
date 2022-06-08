import numpy as np
import cupy as cp
import Parameters as c
#from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat
from scipy.sparse.linalg import spsolve, splu, factorized
from LinearAlgebra import cg
from pyamg import rootnode_solver
import time

class EllipticCpl2:
    def __init__(self, vc, g, c):

        # load appropriate module for working with objects on CPU / GPU
        if c.use_gpu:
            if not c.linear_solver == 'amgx':
                raise ValueError("Invalid solver choice.")
            
            import cupy as xp
            from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat

            areaCell_cpu = g.areaCell.get()
        else:
            if c.linear_solver == 'amgx':
                raise ValueError("Invalid solver choice.")

            import numpy as xp
            from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat

            areaCell_cpu = g.areaCell
        
        # Construct matrix blocks of the coupled elliptic system
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell[:,0], 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        #mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell[:,0].copy( )
            areaCell_psi[g.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            #mAreaCell_psi.eliminate_zeros( )
            
        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi, using the normal vector
        # Left, row 1
        self.AMC = mAreaCell_psi * vc.mVertex2cell * vc.mCurl_t
        self.AC = mAreaCell_psi * vc.mCurl_v
        #self.AMC.eliminate_zeros( )
        self.AMC.sort_indices( )
        #self.AC.eliminate_zeros( )
        self.AC.sort_indices( )
        
        # Left, row 2
        self.AMD = mAreaCell_phi * vc.mVertex2cell * vc.mDiv_t
        self.AD = mAreaCell_phi * vc.mDiv_v
        #self.AMD.eliminate_zeros( )
        self.AMD.sort_indices( )
        #self.AD.eliminate_zeros( )
        self.AD.sort_indices( )
        
        # Right, col 2
        self.GN = vc.mGrad_tn * vc.mCell2vertex_n
        #self.GN.eliminate_zeros( )
        self.GN.sort_indices( )
        
        # Right, col 1
        self.SN = vc.mSkewgrad_nd * vc.mCell2vertex_psi
        #self.SN.eliminate_zeros( )
        self.SN.sort_indices( )

        ## Construct an artificial thickness vector
        thickness_edge = 100 * (10. + xp.random.rand(g.nEdges))
        self.thicknessInv = 1. / thickness_edge
#        self.mThicknessInv = eye(g.nEdges)  
#        self.mThicknessInv.data[0,:] = 1./thickness_edge

        # Copy certain matrices over from VectorCalculus; maybe unnecessary
        # in the future.
        self.mSkewgrad_td = vc.mSkewgrad_td.copy( )
        self.mGrad_n_n = vc.mGrad_n_n.copy( )
        
        
        if c.linear_solver == 'amgx':
            import pyamgx

            pyamgx.initialize( )

            err_tol = c.err_tol*1e-5*np.mean(areaCell_cpu)*np.sqrt(g.nCells)  # For vorticity
            cfg1 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": c.print_stats, 
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
                    "print_solve_stats": c.print_stats, 
                    "obtain_timings": c.print_stats, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            # Smaller error tolerance for divergence because geophysical flows
            # are largely nondivergent
            err_tol = c.err_tol*1e-6*np.mean(areaCell_cpu)*np.sqrt(g.nCells)
            cfg2 = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": c.print_stats, 
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
                    "print_solve_stats": c.print_stats, 
                    "obtain_timings": c.print_stats, 
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
            self.slv11 = pyamgx.Solver().create(rsc1, cfg1, mode)
            self.slv22 = pyamgx.Solver().create(rsc2, cfg2, mode)

            # Create matrices and vectors:
            self.d_A11 = pyamgx.Matrix().create(rsc1, mode)
            self.d_x = pyamgx.Vector().create(rsc1, mode)
            self.d_b1 = pyamgx.Vector().create(rsc1, mode)
            self.d_A22 = pyamgx.Matrix().create(rsc2, mode)
            self.d_y = pyamgx.Vector().create(rsc2, mode)
            self.d_b2 = pyamgx.Vector().create(rsc2, mode)
            
        elif c.linear_solver == 'amg':
            from pyamg import rootnode_solver
        
        elif c.linear_solver == 'lu':
            pass
        
        else:
            raise ValueError("Invalid solver choice.")

    def update(self, thickness_edge, vc, c, g):
        
        self.thicknessInv[:] = 1./thickness_edge
        
        ## Construct the blocks
        self.A11 = self.AC.multiply(self.thicknessInv)
        self.A11 *= self.mSkewgrad_td
        
        self.A12 = self.AMC.multiply(self.thicknessInv)
        self.A12 *= self.mGrad_n_n
        self.A12 += self.AC.multiply(self.thicknessInv) * self.GN
        self.A12 *= 0.5
        
        self.A21 = self.AD.multiply(self.thicknessInv)
        self.A21 *= self.SN
        self.A21 += self.AMD.multiply(self.thicknessInv) * self.mSkewgrad_td
        self.A21 *= 0.5
        
        self.A22 = self.AD.multiply(self.thicknessInv)
        self.A22 *= self.mGrad_n_n

        #self.A11 = self.A11.tolil( )
        #self.A22 = self.A22.tolil( )
        if c.on_a_global_sphere:
            self.A11[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
        else:
            self.A11[g.cellBoundary-1, g.cellBoundary-1] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
        #self.A11 = self.A11.tocsr( )
        #self.A22 = self.A22.tocsr( )
        
        if c.linear_solver == 'lu':
            # Convert the matrices to CSC for better performance
            self.A11 = self.A11.tocsc( )
            self.A22 = self.A22.tocsc( )
            
        elif c.linear_solver == 'amg':
            self.A11 *= -1
            self.A12 *= -1
            self.A21 *= -1
            self.A22 *= -1

            B11 = np.ones((self.A11.shape[0],1), dtype=self.A11.dtype); BH11 = B11.copy()
            self.A11_solver = rootnode_solver(self.A11, B=B11, BH=BH11,
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

            B22 = np.ones((self.A22.shape[0],1), dtype=self.A22.dtype); BH22 = B22.copy()
            self.A22_solver = rootnode_solver(self.A22, B=B22, BH=BH22,
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
            
        elif c.linear_solver == 'amgx':
            self.d_A11.upload_CSR(self.A11)
            self.d_A22.upload_CSR(self.A22)
            self.slv11.setup(self.d_A11)
            self.slv22.setup(self.d_A22)
        
        else:
            raise ValueError("Invalid solver choice.")


    def solve(self, b1, b2, x, y, env=None, nIter = 10):
        
        if c.linear_solver == 'lu':
            A11_lu = splu(self.A11)
            A22_lu = splu(self.A22)
            b1_new = b1.copy(); b2_new = b2.copy( )
            
            for k in np.arange(nIter):
                b1_new[:] = b1 - self.A12.dot(y)
                b2_new[:] = b2 - self.A21.dot(x)
                x[:] = A11_lu.solve(b1_new)
                y[:] = A22_lu.solve(b2_new)

        elif c.linear_solver == 'amgx':

            b1_new = b1.copy(); b2_new = b2.copy( )
            for k in np.arange(nIter):
                b1_new[:] = b1 - self.A12.dot(y)
                b2_new[:] = b2 - self.A21.dot(x)
                
                self.d_b1.upload(b1_new)
                self.d_b2.upload(b2_new)
                self.d_x.upload(x)
                self.d_y.upload(y)
                self.slv11.solve(self.d_b1, self.d_x)
                self.slv22.solve(self.d_b2, self.d_y)
                self.d_x.download_raw(x.data)
                self.d_y.download_raw(y.data)
                

        elif c.linear_solver == 'amg':
            x_res = []; y_res = []
            for k in np.arange(nIter):
                b11 = b1 - self.A12.dot(y)
                b22 = b2 - self.A21.dot(x)
                x[:] = self.A11_solver.solve(b11, x0=x, tol=c.err_tol, residuals=x_res)
                y[:] = self.A22_solver.solve(b22, x0=y, tol=c.err_tol, residuals=y_res)
                if c.print_stats:
                    print("k, #iter A11, #iter A22: %d, %d, %d" % (k, len(x_res), len(y_res)))

            # Negate the solution, since the matrices were negated in
            # the update stage for positive definiteness
            x *= -1; y *= -1 
            
        else:
            raise ValueError("Invalid solver choice.")
        

class Poisson:
    def __init__(self, vc, g, c):

        # load appropriate module for working with objects on CPU / GPU
        if c.use_gpu:
            if not c.linear_solver == 'amgx':
                raise ValueError("Invalid solver choice.")
            
            import cupy as xp
            from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat

            areaCell_cpu = g.areaCell.get()
        else:
            if c.linear_solver == 'amgx':
                raise ValueError("Invalid solver choice.")

            import numpy as xp
            from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat

            areaCell_cpu = g.areaCell

        #
        # Laplace on primal (voronoi)
        #
        ## Construct the matrix representing the discrete Laplace operator the primal
        ## mesh (Voronoi mesh). Homogeneous Neuman BC's are assumed.
        from swe_comp import swe_comp as cmp
        nEntries, rows, cols, valEntries = \
            cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                                                  areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nCells))
        self.A = A.tocsr( )

        if c.on_a_global_sphere:
            mAreaCell = diags(g.areaCell[:,0], 0, format='csr')
        else:
            raise ValueError('Bounded domains are not supported at this time.')

        # Scale matrix A to make it symmetric
        self.A = mAreaCell * self.A

        if c.linear_solver == 'lu':
            A = self.A.tocsc( )
            self.lu = splu(A)
            
        elif c.linear_solver == 'cg':
            # Nothing to do
            pass
            
        elif c.linear_solver == 'pcg':
            A_t = -self.D2s
            
        elif c.linear_solver == 'amg':

            self.A_spd = self.A.copy( )
            self.A_spd = -self.A_spd
            self.B = np.ones((self.A_spd.shape[0],1), dtype=self.A_spd.dtype); self.BH = self.B.copy()

            if self.A_spd.shape[0] in  [40962, 163842, 655362]:
                self.A_amg = rootnode_solver(self.A_spd, B=self.B, BH=self.BH,
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
            elif self.A_spd.shape[0] in [81920, 327680, 1310720, 5242880]:
                self.A_amg = rootnode_solver(self.A_spd, B=self.B, BH=self.BH,
                    strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
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
            elif self.A_spd.shape[0] in  [2621442]:
                self.A_amg = rootnode_solver(self.A_spd, B=self.B, BH=self.BH,
                    strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
                    smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 3, 'maxiter': 4}),
                    improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), \
                                        None, None, None, None, None, None, None, None, None, None, \
                                        None, None, None, None],
                    aggregate="standard",
                    presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                    postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                    max_levels=15,
                    max_coarse=300,
                    coarse_solver="pinv")
                
            else:
                print("Unknown matrix. Using a generic AMG solver")

                self.A_amg = rootnode_solver(self.A_spd, B=self.B, BH=self.BH,
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

        elif c.linear_solver == 'amgx':
            import pyamgx

            pyamgx.initialize( )

            err_tol = c.err_tol*1e-6*np.mean(areaCell_cpu)*np.sqrt(g.nCells)  # For vorticity
            #err_tol = c.err_tol
            cfg = pyamgx.Config( ).create_from_dict({    
                "config_version": 2, 
                "determinism_flag": 0, 
                "solver": {
                    "preconditioner": {
                        "print_grid_stats": c.print_stats, 
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
                    "print_solve_stats": c.print_stats, 
                    "obtain_timings": c.print_stats, 
                    "max_iters": c.max_iters, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "tolerance": err_tol,
                    "norm": "L2"
                }
            })

            rsc = pyamgx.Resources().create_simple(cfg)
            mode = 'dDDI'
            
            # Create solver:
            self.amgx = pyamgx.Solver().create(rsc, cfg, mode)

            # Create matrices and vectors:
            d_A = pyamgx.Matrix().create(rsc, mode)
            self.d_x = pyamgx.Vector().create(rsc, mode)
            self.d_b = pyamgx.Vector().create(rsc, mode)

            d_A.upload_CSR(self.A)

            # Setup and solve system:
            self.amgx.setup(d_A)

        else:
            raise ValueError("Invalid solver choice.")

    def solve(self, b, x):
        
        if c.linear_solver == 'lu':
            x[:] = self.lu.solve(b)
        elif c.linear_solver == 'cg':
            try:
                info, nIter = cg(self.A, b, x, max_iter=1000, relres=1e-9)
                print("CG, nIter = %d" % nIter)
            except KeyError:
                raise KeyError
                
        elif c.linear_solver == 'amg':
            
            res = []
            x0 = -x
#            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel='cg', maxiter=300, cycle='V')
#            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel='cg')
            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res)
            x *= -1.
            
            print("AMG, nIter = %d" % (len(res),))

        elif c.linear_solver == 'amgx':
#            b_cp = b.copy()
#            x_cp = x.copy()
            self.d_b.upload(b)
            self.d_x.upload(x)
            self.amgx.solve(self.d_b, self.d_x)
            self.d_x.download_raw(x.data)
#            x[:] = x_cp[:]

            ### DEBUGGING
#            print("Inside solve")
#            print("b print out")
#            print(b[0:10])
#            print("b max: %e" % (b.max()))
#            print("b min: %e" % (b.min()))
#            print("x printout")
#            print(x[:10])
#            if x[1] > -1e7 and x[1] < -1e6:
#                raise ValueError("Stop for checking")
            
        else:
            raise ValueError("Invalid solver choice.")
        
        
