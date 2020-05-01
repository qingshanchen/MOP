import numpy as np
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat
from scipy.sparse.linalg import spsolve, splu, factorized
from swe_comp import swe_comp as cmp
from LinearAlgebra import cg
from pyamg import rootnode_solver
import time

class EllipticCPL:
    def __init__(self, A, linear_solver, env):


        if linear_solver is 'lu':
            self.A = A.tocsc( )
            
        elif linear_solver is 'amgx':
            import pyamgx

            pyamgx.initialize( )

            hA = A.tocsr( )
            AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json'

            if False:
                cfg = pyamgx.Config( ).create_from_file(AMGX_CONFIG_FILE_NAME)
            else:
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
                            "max_levels": 1000, 
                            "postsweeps": 2, 
                            "cycle": "V"
                        }, 
                        "solver": "PCGF", 
                        "print_solve_stats": c.print_stats, 
                        "obtain_timings": c.print_stats, 
                        "max_iters": c.max_iters, 
                        "monitor_residual": 1, 
                        "convergence": "RELATIVE_INI", 
                        "scope": "main", 
                        "tolerance": c.err_tol, 
                        "norm": "L2"
                    }
                })
                
            rsc = pyamgx.Resources().create_simple(cfg)
            mode = 'dDDI'

            # Create solver:
            self.amgx = pyamgx.Solver().create(rsc, cfg, mode)

            # Create matrices and vectors:
            self.d_A = pyamgx.Matrix().create(rsc, mode)
            self.d_x = pyamgx.Vector().create(rsc, mode)
            self.d_b = pyamgx.Vector().create(rsc, mode)

            self.d_A.upload_CSR(hA)

            # Setup and solve system:
            # self.amgx.setup(d_A)

            ## Clean up:
            #A.destroy()
            #x.destroy()
            #b.destroy()
            #self.amgx.destroy()
            #rsc.destroy()
            #cfg.destroy()

            #pyamgx.finalize()

        elif linear_solver is 'cg' or 'amg':
            pass
        
        else:
            raise ValueError("Invalid solver choice.")

    def solve(self, A, b, x, env=None, linear_solver='lu'):
        
        if linear_solver is 'lu':
            x[:] = spsolve(A, b)

        elif linear_solver is 'amgx':
            self.d_b.upload(b)
            self.d_x.upload(x)
            #self.d_A.replace_coefficients(A.data)
            self.d_A.upload_CSR(A)
            self.amgx.setup(self.d_A)
            self.amgx.solve(self.d_b, self.d_x)
            self.d_x.download(x)

        elif linear_solver is 'cg':
            t0 = time.clock()
            t0a = time.time( )
            err, counter = cg(env, A, b, x, max_iter = c.max_iters, relres = c.err_tol)
            t1 = time.clock()
            t1a = time.time( )
            if c.print_stats:
                print("CG # iters, cpu time, wall time: %d %f %f" % (counter, t1-t0, t1a-t0a))

        else:
            raise ValueError("Invalid solver choice.")


class EllipticCpl2:
    def __init__(self, mVertex2cell, mCurl_t, mCurl_v, mDiv_t, mDiv_v, \
                 mGrad_tn, mCell2vertex_n, mSkewgrad_nd, mCell2vertex_psi, g, c):

        # Construct matrix blocks of the coupled elliptic system
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell, 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[g.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            mAreaCell_psi.eliminate_zeros( )
            
        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi, using the normal vector
        # Left, row 1
        self.AMC = mAreaCell_psi * mVertex2cell * mCurl_t
        self.AC = mAreaCell_psi * mCurl_v
        self.AMC.eliminate_zeros( )
        self.AC.eliminate_zeros( )
        
        # Left, row 2
        self.AMD = mAreaCell_phi * mVertex2cell * mDiv_t
        self.AD = mAreaCell_phi * mDiv_v
        self.AMD.eliminate_zeros( )
        self.AD.eliminate_zeros( )
        
        # Right, col 2
        self.GN = mGrad_tn * mCell2vertex_n
        self.GN.eliminate_zeros( )
        
        # Right, col 1
        self.SN = mSkewgrad_nd * mCell2vertex_psi
        self.SN.eliminate_zeros( )
        
        ## Construct an artificial thickness vector
        thickness_edge = 100 * (10. + np.random.rand(g.nEdges))
        self.mThicknessInv = eye(g.nEdges)  
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        if c.linear_solver is 'amgx':
            raise ValueError("Not ready yet for this solver")
            import pyamgx

            pyamgx.initialize( )

            err_tol = c.err_tol*1e-5*np.mean(g.areaCell)*np.sqrt(g.nCells)  # For vorticity
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
            err_tol = c.err_tol*1e-6*np.mean(g.areaCell)*np.sqrt(g.nCells)
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
            
        elif c.linear_solver is 'amg':
            from pyamg import rootnode_solver
        
        else:
            raise ValueError("Invalid solver choice.")

    def update(self, thickness_edge, mSkewgrad_td, mGrad_n_n, c, g):
        self.mThicknessInv.data[0,:] = 1./thickness_edge

        ## Construct the blocks
        self.A11 = self.AC * self.mThicknessInv * mSkewgrad_td
        self.A12 = self.AMC * self.mThicknessInv * mGrad_n_n
        self.A12 += self.AC * self.mThicknessInv * self.GN
        self.A12 *= 0.5
        self.A21 = self.AD * self.mThicknessInv * self.SN
        self.A21 += self.AMD * self.mThicknessInv * mSkewgrad_td
        self.A21 *= 0.5
        self.A22 = self.AD * self.mThicknessInv * mGrad_n_n

        if c.on_a_global_sphere:
            self.A11[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
        else:
            self.A11[g.cellBoundary-1, g.cellBoundary-1] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
        
        if c.linear_solver is 'lu':
            raise ValueError("Not ready yet for this solver")
        
            # Convert the matrices to CSC for better performance
            self.A11.tocsc( )
            self.A22.tocsc( )
            
        elif c.linear_solver is 'amg':
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
            
        elif c.linear_solver is 'amgx':
            raise ValueError("Not ready yet for this solver")
            

        
        else:
            raise ValueError("Invalid solver choice.")


    def solve(self, b1, b2, x, y, env=None, nIter = 10):
        
        if c.linear_solver is 'lu':
            raise ValueError("Not ready yet for this solver")

        elif c.linear_solver is 'amgx':
            raise ValueError("Not ready yet for this solver")

        elif c.linear_solver is 'amg':
            x_res = []; y_res = []
            x_tmp = x; y_tmp = y
            for k in np.arange(nIter):
                b11 = b1 - self.A12.dot(y_tmp)
                b22 = b2 - self.A21.dot(x_tmp)
                x_tmp = self.A11_solver.solve(b11, x0=x_tmp, tol=c.err_tol, residuals=x_res)
                y_tmp = self.A22_solver.solve(b22, x0=y_tmp, tol=c.err_tol, residuals=y_res)
                print("k = %d,  AMG nIters = %d, %d" % (k, len(x_res), len(y_res)))
                print(x_res)
                print(y_res)

            # Negate the solution, since the matrices were negated in
            # the update stage for positive definiteness
            x[:] = -1 * x_tmp; y[:] = -1 * y_tmp
            
        else:
            raise ValueError("Invalid solver choice.")
        

class Poisson:
    def __init__(self, A, linear_solver, env):

        if linear_solver is 'lu':
            self.A = A.tocsc( )
            self.lu = splu(self.A)
        elif linear_solver is 'cg':
            self.A = A.tocsr( )
        elif linear_solver in ['cudaCG']:
            self.A = A.tocsr( )
            self.dData = env.cuda.to_device(self.A.data)
            self.dPtr = env.cuda.to_device(self.A.indptr)
            self.dInd = env.cuda.to_device(self.A.indices)
            self.cuSparseDescr = env.cuSparse.matdescr( )
        elif linear_solver in ['cudaPCG']:
            self.A = A.tocsr( )
            self.Adescr = env.cuSparse.matdescr( )
            self.Adata = env.cuda.to_device(self.A.data)
            self.Aptr = env.cuda.to_device(self.A.indptr)
            self.Aind = env.cuda.to_device(self.A.indices)

            A_t = self.A.copy( )
            A_t = -A_t    # Make it positive definite
            A_t.data = np.where(A_t.nonzero()[0] >= A_t.nonzero()[1], A_t.data, 0.)
            A_t.eliminate_zeros( )
            A_t_descr = env.cuSparse.matdescr(matrixtype='S', fillmode='L')
            info = env.cuSparse.csrsv_analysis(trans='N', m=A_t.shape[0], nnz=A_t.nnz, \
                                       descr=A_t_descr, csrVal=A_t.data, \
                                       csrRowPtr=A_t.indptr, csrColInd=A_t.indices)
            env.cuSparse.csric0(trans='N', m=A_t.shape[0], \
                        descr=A_t_descr, csrValM=A_t.data, csrRowPtrA=A_t.indptr,\
                        csrColIndA=A_t.indices, info=info)

            self.L = A_t
            self.Lmv_descr = env.cuSparse.matdescr( )
    #        self.Lsv_descr = cuSparse.matdescr(matrixtype='T', fillmode='L')
            self.Lsv_descr = env.cuSparse.matdescr(matrixtype='T')
            self.Ldata = env.cuda.to_device(self.L.data)
            self.Lptr = env.cuda.to_device(self.L.indptr)
            self.Lind = env.cuda.to_device(self.L.indices)
            self.Lsv_info = env.cuSparse.csrsv_analysis(trans='N', m=self.L.shape[0], \
                    nnz=self.L.nnz,  descr=self.Lsv_descr, csrVal=self.Ldata, \
                    csrRowPtr=self.Lptr, csrColInd=self.Lind)        


            self.LT = self.L.transpose( )
            self.LT.tocsr( )
            self.LTmv_descr = env.cuSparse.matdescr( )
#            self.LTsv_descr = env.cuSparse.matdescr(matrixtype='T', fillmode='U')
            self.LTsv_descr = env.cuSparse.matdescr()
            self.LTdata = env.cuda.to_device(self.LT.data)
            self.LTptr = env.cuda.to_device(self.LT.indptr)
            self.LTind = env.cuda.to_device(self.LT.indices)
            self.LTsv_info = env.cuSparse.csrsv_analysis(trans='T', m=self.L.shape[0], \
                    nnz=self.L.nnz,  descr=self.Lsv_descr, csrVal=self.Ldata, \
                    csrRowPtr=self.Lptr, csrColInd=self.Lind)        

        elif linear_solver is 'pcg':
            self.A = A.tocsr( )
            A_t = -self.D2s

            
        elif linear_solver is 'amg':

            self.A = A.tocsr( )
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

        elif linear_solver is 'amgx':
            import pyamgx

            pyamgx.initialize( )

            hA = A.tocsr( )
            if hA.nnz * 1. / hA.shape[0] > 5.5:           # Primary mesh
                AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json'
            if hA.nnz * 1. / hA.shape[0] < 5.5:           # Dual mesh
                AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json'
            else:
                print('Error: cannot determine primary or dual mesh, not sure which config to use.')


            cfg = pyamgx.Config( ).create_from_file(AMGX_CONFIG_FILE_NAME) 
            rsc = pyamgx.Resources().create_simple(cfg)
            mode = 'dDDI'

            # Create solver:
            self.amgx = pyamgx.Solver().create(rsc, cfg, mode)

            # Create matrices and vectors:
            d_A = pyamgx.Matrix().create(rsc, mode)
            self.d_x = pyamgx.Vector().create(rsc, mode)
            self.d_b = pyamgx.Vector().create(rsc, mode)

            d_A.upload(hA.indptr, hA.indices, hA.data)

            # Setup and solve system:
            self.amgx.setup(d_A)

            ## Clean up:
            #A.destroy()
            #x.destroy()
            #b.destroy()
            #self.amgx.destroy()
            #rsc.destroy()
            #cfg.destroy()

            #pyamgx.finalize()
        else:
            raise ValueError("Invalid solver choice.")

    def solve(self, b, x, env=None, linear_solver='lu'):
        
        if linear_solver is 'lu':
            x[:] = self.lu.solve(b)
        elif linear_solver is 'cg':
            try:
                info, nIter = cg(env, self.A, b, x, max_iter=c.max_iter, relres=c.err_tol)
                print("CG, nIter = %d" % nIter)
            except KeyError:
                raise KeyError
                
        elif linear_solver is 'cudaCG':
            info, nIter = cudaCG(env, self, b, x, max_iter=c.max_iter, relres = c.err_tol)
            print("cudaCG, nIter = %d" % nIter)
        elif linear_solver is 'cudaPCG':
            info, nIter = cudaPCG(env, self, b, x, max_iter=c.max_iter, relres = c.err_tol)
            print("cudaPCG, nIter = %d" % nIter)
        elif linear_solver is 'amg':
            
            res = []
            x0 = -x
#            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel='cg', maxiter=300, cycle='V')
#            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel='cg')
            x[:] = self.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res)
            x *= -1.
            
            print("AMG, nIter = %d" % (len(res),))

        elif linear_solver is 'amgx':
            self.d_b.upload(b)
            self.d_x.upload(x)
            self.amgx.solve(self.d_b, self.d_x)
            self.d_x.download(x)
        else:
            raise ValueError("Invalid solver choice.")
        
        
class Device_CSR:
    def __init__(self, A, env):
        self.dData = env.cuda.to_device(A.data)
        self.dPtr = env.cuda.to_device(A.indptr)
        self.dInd = env.cuda.to_device(A.indices)
        self.shape = A.shape
        self.nnz = A.nnz
        self.cuSparseDescr = env.cuSparse.matdescr( )
#        self.d_vectOut = env.cuda.device_array


class VectorCalculus:
    def __init__(self, g, c, env):
        self.env = env

        self.linear_solver = c.linear_solver

        self.max_iters = c.max_iters
        self.err_tol = c.err_tol

        self.areaCell = g.areaCell.copy()
        self.areaTriangle = g.areaTriangle.copy()

        #
        # Mesh element indices; These should be in the grid object(?)
        #
        if not c.on_a_global_sphere:
            # Collect non-boundary (interior) cells and put into a vector,
            # and boundary cells into a separate vector
            nCellsBoundary = np.sum(g.boundaryCellMark[:]>0)
            nCellsInterior = g.nCells - nCellsBoundary
            
            self.cellInterior, self.cellBoundary, self.cellRankInterior, \
                cellInner_tmp, cellOuter_tmp, self.cellRankInner, \
                nCellsInner, nCellsOuter = \
                cmp.separate_boundary_interior_inner_cells(nCellsInterior,  \
                nCellsBoundary, c.max_int, g.boundaryCellMark, g.cellsOnCell, g.nEdgesOnCell)
            self.cellInner = cellInner_tmp[:nCellsInner]
            self.cellOuter = cellOuter_tmp[:nCellsOuter]

            self.cellBoundary_ord = cmp.boundary_cells_ordered(\
                                nCellsBoundary, g.boundaryCellMark, g.cellsOnCell)

        else:
            self.cellBoundary = np.array([], dtype='int')

        #
        # Divergence on primal
        #
        # Construct the matrix representing the discrete div on the primal mesh (Voronoi cells)
        # No-flux BCs assumed on the boundary
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mDiv_v = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_v = Device_CSR(self.mDiv_v, env)

        #
        # Divergence on dual (triangle)
        #
        ## Construct the matrix representing the discrete div on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mDiv_t = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_t = Device_CSR(self.mDiv_t, env)

        #
        # Curl on primal
        #
        ## Construct the matrix representing the discrete curl on the primal mesh (Voronoi cells)
        ## No-slip BCs assumed on the boundary.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mCurl_v = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_v = Device_CSR(self.mCurl_v, env)

        #
        # Curl on dual
        #
        ## Construct the matrix representing the discrete curl on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mCurl_t = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_t = Device_CSR(self.mCurl_t, env)

        #
        # Laplace on primal (voronoi)
        #
        ## Construct the matrix representing the discrete Laplace operator the primal
        ## mesh (Voronoi mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, \
                                                  g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        self.mLaplace_v = A.tocsr( )

        if c.use_gpu:
            self.d_mLaplace_v = Device_CSR(self.mLaplace_v, env)

        #
        # Laplace on dual (triangle)
        #
        ## Construct the matrix representing the discrete Laplace operator the dual
        ## mesh (triangular mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, \
                                       g.verticesOnEdge, g.dvEdge, g.dcEdge, g.areaTriangle)
                                                  
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        self.mLaplace_t = A.tocsr( )

        if c.use_gpu:
            self.d_mLaplace_t = Device_CSR(self.mLaplace_t, env)

        #
        # Gradient normal
        #
        ## Construct the matrix representing the discrete grad operator along the normal direction.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mGrad_n = A.tocsr( )

        A_n = A.tolil()   
        A_n[:,0] = 0.
        self.mGrad_n_n = A_n.tocsr( )
        self.mGrad_n_n.eliminate_zeros()

        if c.use_gpu:
            self.d_mGrad_n = Device_CSR(self.mGrad_n, env)

        #
        # Gradient tangential(?) with Dirichlet
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Dirichlet BC's 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_td(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_td = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_td = Device_CSR(self.mGrad_td, env)

        #
        # Gradient tangential(?) with Neumann
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Neumann BC's
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_tn(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_tn = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_tn = Device_CSR(self.mGrad_tn, env)

        #
        # Skew gradient tangential
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction.  mSkewgrad_t = mGrad_n
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_t(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_t = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_t = Device_CSR(self.mSkewgrad_t, env)

        #
        # Skew gradient tangential w. Dirichlet
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction. Homogeneous Dirichlet assumed
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_td(g.cellsOnEdge, g.dcEdge, g.boundaryCellMark)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_td = A.tocsr( )
        self.mSkewgrad_td.eliminate_zeros( )
        
        if c.use_gpu:
            self.d_mSkewgrad_td = Device_CSR(self.mSkewgrad_td, env)

        #
        # Skew gradient normal w. Dirichlet
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the normal direction. Homogeneous Dirichlet assumed.
        ## mSkewgrad_n = - mGrad_td
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_nd(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mSkewgrad_nd = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_nd = Device_CSR(self.mSkewgrad_nd, env)

        #
        # Map from cell to vertex
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex = Device_CSR(self.mCell2vertex, env)

        A_n = A.tolil( )
        A_n[:,0] = 0.       # zero for entry 0; Neumann
        self.mCell2vertex_n = A_n.tocsr()
        self.mCell2vertex_n.eliminate_zeros( )

        #
        # Map cell to vertex w. Dirichlet
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh; homogeneous Dirichlet BC's are assumed
        ## On a global sphere, cell 0 is considered the single boundary pt.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex_psi(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.boundaryCellMark, c.on_a_global_sphere)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex_psi = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex_psi = Device_CSR(self.mCell2vertex_psi, env)

        #
        # Map vertex to cell
        #
        ## Construct the matrix representing the mapping from the dual mesh onto the primal
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_vertex2cell(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nVertices))
        self.mVertex2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mVertex2cell = Device_CSR(self.mVertex2cell, env)

        #
        # Map cell to edge
        #
        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2edge = Device_CSR(self.mCell2edge, env)

        #
        # Map edge to cell
        # 
        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mEdge2cell = Device_CSR(self.mEdge2cell, env)

        ## Construct an artificial thickness vector
        thickness_edge = 100 * (10. + np.random.rand(g.nEdges))
        self.mThicknessInv = eye(g.nEdges)  
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
#        if c.use_gpu:                        # Need to update at every step
#            d_mThicknessInv = Device_CSR(self.mThicknessInv.to_csr(), env)

        # Construct the coefficient matrix for the coupled elliptic problem
        self.AC, self.AMC, self.AMD, self.AD, self.GN, self.SN = \
            self.construct_EllipticCPL_blocks(env, g, c)
        
        self.coefM = None
        self.update_matrix_for_coupled_elliptic(thickness_edge, c, g)

        self.POcpl = EllipticCPL(self.coefM, c.linear_solver, env)

        # Construct the coefficient matrices for the coupled elliptic problem
        self.A11 = None
        self.A12 = None
        self.A21 = None
        self.A22 = None
        self.update_matrices_for_coupled_elliptic(thickness_edge, c, g)

        # Construct the EllipticCpl2 object for the coupled elliptic system
        self.POcpl2 = EllipticCpl2(self.mVertex2cell, self.mCurl_t, self.mCurl_v, \
                self.mDiv_t, self.mDiv_v, self.mGrad_tn, self.mCell2vertex_n, \
                                   self.mSkewgrad_nd, self.mCell2vertex_psi, g, c)
        
        ## Some temporary variables as place holders
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                    g.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*g.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        D2s_coo.eliminate_zeros( )

        self.POpn = Poisson(D2s_coo, self.linear_solver, env)
            
    def construct_EllipticCPL_blocks(self, env, g, c):
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell, 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[self.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            mAreaCell_psi.eliminate_zeros( )
            
        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi, using the normal vector
        # Left, row 1
        AMC = mAreaCell_psi * self.mVertex2cell * self.mCurl_t
        AC = mAreaCell_psi * self.mCurl_v
        AMC.eliminate_zeros( )
        AC.eliminate_zeros( )
        
        # Left, row 2
        AMD = mAreaCell_phi * self.mVertex2cell * self.mDiv_t
        AD = mAreaCell_phi * self.mDiv_v
        AMD.eliminate_zeros( )
        AD.eliminate_zeros( )
        
        # Right, col 2
        GN = self.mGrad_tn * self.mCell2vertex_n
        GN.eliminate_zeros( )
        
        # Right, col 1
        SN = self.mSkewgrad_nd * self.mCell2vertex_psi
        SN.eliminate_zeros( )
        
        return AC, AMC, AMD, AD, GN, SN

    
    def update_matrix_for_coupled_elliptic(self, thickness_edge, c, g):
        self.mThicknessInv.data[0,:] = 1./thickness_edge

        ## Construct the blocks
        A11 = self.AC * self.mThicknessInv * self.mSkewgrad_td
        A12 = self.AMC * self.mThicknessInv * self.mGrad_n_n
        A12 += self.AC * self.mThicknessInv * self.GN
        A12 *= 0.5
        A21 = self.AD * self.mThicknessInv * self.SN
        A21 += self.AMD * self.mThicknessInv * self.mSkewgrad_td
        A21 *= 0.5
        A22 = self.AD * self.mThicknessInv * self.mGrad_n_n

        self.coefM = bmat([[A11, A12], [A21, A22]], format = 'csr')

        if c.on_a_global_sphere:
            self.coefM[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
            self.coefM[g.nCells, g.nCells] = -2*np.sqrt(3.)/thickness_edge[0]
        else:
            self.coefM[self.cellBoundary-1, self.cellBoundary-1] = -2*np.sqrt(3.)/thickness_edge[0]
            self.coefM[g.nCells, g.nCells] = -2*np.sqrt(3.)/thickness_edge[0]


    def update_matrices_for_coupled_elliptic(self, thickness_edge, c, g):
        self.mThicknessInv.data[0,:] = 1./thickness_edge

        ## Construct the blocks
        self.A11 = self.AC * self.mThicknessInv * self.mSkewgrad_td
        self.A12 = self.AMC * self.mThicknessInv * self.mGrad_n_n
        self.A12 += self.AC * self.mThicknessInv * self.GN
        self.A12 *= 0.5
        self.A21 = self.AD * self.mThicknessInv * self.SN
        self.A21 += self.AMD * self.mThicknessInv * self.mSkewgrad_td
        self.A21 *= 0.5
        self.A22 = self.AD * self.mThicknessInv * self.mGrad_n_n

#        print("Inside update_matrices, thickness_edge[0] = %f" % thickness_edge[0])

        if c.on_a_global_sphere:
            self.A11[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]
        else:
            self.A11[self.cellBoundary-1, self.cellBoundary-1] = -2*np.sqrt(3.)/thickness_edge[0]
            self.A22[0,0] = -2*np.sqrt(3.)/thickness_edge[0]

#        print("At end of update_matrices, A11[0,0] = %f" % self.A11[0,0])
#        return A11, A12, A21, A22
    
            
    def discrete_div_v(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mDiv_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mDiv_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mDiv_v.shape[0], \
                n=self.d_mDiv_v.shape[1], nnz=self.d_mDiv_v.nnz, alpha=1.0, \
                descr=self.d_mDiv_v.cuSparseDescr, csrVal=self.d_mDiv_v.dData, \
                csrRowPtr=self.d_mDiv_v.dPtr, csrColInd=self.d_mDiv_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mDiv_v.dot(vEdge)


    def discrete_div_t(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mDiv_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mDiv_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mDiv_t.shape[0], \
                n=self.d_mDiv_t.shape[1], nnz=self.d_mDiv_t.nnz, alpha=1.0, \
                descr=self.d_mDiv_t.cuSparseDescr, csrVal=self.d_mDiv_t.dData, \
                csrRowPtr=self.d_mDiv_t.dPtr, csrColInd=self.d_mDiv_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mDiv_t.dot(vEdge)
        

    def discrete_curl_v(self, vEdge):
        '''
        The discrete curl operator on the primal mesh.
        No-slip boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl_v.shape[0], \
                n=self.d_mCurl_v.shape[1], nnz=self.d_mCurl_v.nnz, alpha=1.0, \
                descr=self.d_mCurl_v.cuSparseDescr, csrVal=self.d_mCurl_v.dData, \
                csrRowPtr=self.d_mCurl_v.dPtr, csrColInd=self.d_mCurl_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl_v.dot(vEdge)


    def discrete_curl_t(self, vEdge):
        '''
        The discrete curl operator on the dual mesh.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl_t.shape[0], \
                n=self.d_mCurl_t.shape[1], nnz=self.d_mCurl_t.nnz, alpha=1.0, \
                descr=self.d_mCurl_t.cuSparseDescr, csrVal=self.d_mCurl_t.dData, \
                csrRowPtr=self.d_mCurl_t.dPtr, csrColInd=self.d_mCurl_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl_t.dot(vEdge)
        

    def discrete_laplace_v(self, sCell):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        if c.use_gpu:
            assert len(sCell) == self.d_mLaplace_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mLaplace_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mLaplace_v.shape[0], \
                n=self.d_mLaplace_v.shape[1], nnz=self.d_mLaplace_v.nnz, alpha=1.0, \
                descr=self.d_mLaplace_v.cuSparseDescr, csrVal=self.d_mLaplace_v.dData, \
                csrRowPtr=self.d_mLaplace_v.dPtr, csrColInd=self.d_mLaplace_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mLaplace_v.dot(sCell)


    def discrete_laplace_t(self, sVertex):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        if c.use_gpu:
            assert len(sVertex) == self.d_mLaplace_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mLaplace_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mLaplace_t.shape[0], \
                n=self.d_mLaplace_t.shape[1], nnz=self.d_mLaplace_t.nnz, alpha=1.0, \
                descr=self.d_mLaplace_t.cuSparseDescr, csrVal=self.d_mLaplace_t.dData, \
                csrRowPtr=self.d_mLaplace_t.dPtr, csrColInd=self.d_mLaplace_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mLaplace_t.dot(sVertex)

        
    # The discrete gradient operator along the normal direction
    def discrete_grad_n(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mGrad_n.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mGrad_n.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_n.shape[0], \
                n=self.d_mGrad_n.shape[1], nnz=self.d_mGrad_n.nnz, alpha=1.0, \
                descr=self.d_mGrad_n.cuSparseDescr, csrVal=self.d_mGrad_n.dData, \
                csrRowPtr=self.d_mGrad_n.dPtr, csrColInd=self.d_mGrad_n.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_n.dot(sCell)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_grad_td(self, sVertex):
        '''With implied Dirichlet BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mGrad_td.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mGrad_td.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_td.shape[0], \
                n=self.d_mGrad_td.shape[1], nnz=self.d_mGrad_td.nnz, alpha=1.0, \
                descr=self.d_mGrad_td.cuSparseDescr, csrVal=self.d_mGrad_td.dData, \
                csrRowPtr=self.d_mGrad_td.dPtr, csrColInd=self.d_mGrad_td.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_td.dot(sVertex)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Neumann BC's
    def discrete_grad_tn(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mGrad_tn.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mGrad_tn.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_tn.shape[0], \
                n=self.d_mGrad_tn.shape[1], nnz=self.d_mGrad_tn.nnz, alpha=1.0, \
                descr=self.d_mGrad_tn.cuSparseDescr, csrVal=self.d_mGrad_tn.dData, \
                csrRowPtr=self.d_mGrad_tn.dPtr, csrColInd=self.d_mGrad_tn.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_tn.dot(sVertex)


    # The discrete skew gradient operator along the normal direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_skewgrad_nd(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mSkewgrad_nd.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mSkewgrad_nd.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mSkewgrad_nd.shape[0], \
                n=self.d_mSkewgrad_nd.shape[1], nnz=self.d_mSkewgrad_nd.nnz, alpha=1.0, \
                descr=self.d_mSkewgrad_nd.cuSparseDescr, csrVal=self.d_mSkewgrad_nd.dData, \
                csrRowPtr=self.d_mSkewgrad_nd.dPtr, csrColInd=self.d_mSkewgrad_nd.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mSkewgrad_nd.dot(sVertex)

    # The discrete skew gradient operator along the tangential direction
    def discrete_skewgrad_t(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mSkewgrad_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mSkewgrad_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mSkewgrad_t.shape[0], \
                n=self.d_mSkewgrad_t.shape[1], nnz=self.d_mSkewgrad_t.nnz, alpha=1.0, \
                descr=self.d_mSkewgrad_t.cuSparseDescr, csrVal=self.d_mSkewgrad_t.dData, \
                csrRowPtr=self.d_mSkewgrad_t.dPtr, csrColInd=self.d_mSkewgrad_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mSkewgrad_t.dot(sCell)
        

    def cell2vertex(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mCell2vertex.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mCell2vertex.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCell2vertex.shape[0], \
                n=self.d_mCell2vertex.shape[1], nnz=self.d_mCell2vertex.nnz, alpha=1.0, \
                descr=self.d_mCell2vertex.cuSparseDescr, csrVal=self.d_mCell2vertex.dData, \
                csrRowPtr=self.d_mCell2vertex.dPtr, csrColInd=self.d_mCell2vertex.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mCell2vertex.dot(sCell)


    def vertex2cell(self, sVertex):

        if c.use_gpu:
            assert len(sVertex) == self.d_mVertex2cell.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mVertex2cell.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mVertex2cell.shape[0], \
                n=self.d_mVertex2cell.shape[1], nnz=self.d_mVertex2cell.nnz, alpha=1.0, \
                descr=self.d_mVertex2cell.cuSparseDescr, csrVal=self.d_mVertex2cell.dData, \
                csrRowPtr=self.d_mVertex2cell.dPtr, csrColInd=self.d_mVertex2cell.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mVertex2cell.dot(sVertex)
        
    def cell2edge(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mCell2edge.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mCell2edge.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCell2edge.shape[0], \
                n=self.d_mCell2edge.shape[1], nnz=self.d_mCell2edge.nnz, alpha=1.0, \
                descr=self.d_mCell2edge.cuSparseDescr, csrVal=self.d_mCell2edge.dData, \
                csrRowPtr=self.d_mCell2edge.dPtr, csrColInd=self.d_mCell2edge.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mCell2edge.dot(sCell)

    def edge2cell(self, sEdge):

        if c.use_gpu:
            assert len(sEdge) == self.d_mEdge2cell.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sEdge)

            vOut = np.zeros(self.d_mEdge2cell.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mEdge2cell.shape[0], \
                n=self.d_mEdge2cell.shape[1], nnz=self.d_mEdge2cell.nnz, alpha=1.0, \
                descr=self.d_mEdge2cell.cuSparseDescr, csrVal=self.d_mEdge2cell.dData, \
                csrRowPtr=self.d_mEdge2cell.dPtr, csrColInd=self.d_mEdge2cell.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mEdge2cell.dot(sEdge)
