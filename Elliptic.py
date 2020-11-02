import numpy as np
import cupy as cp
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat
from scipy.sparse.linalg import spsolve, splu, factorized
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
    def __init__(self, vc, g, c):

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
        self.AMC = mAreaCell_psi * vc.mVertex2cell * vc.mCurl_t
        self.AC = mAreaCell_psi * vc.mCurl_v
        self.AMC.eliminate_zeros( )
        self.AMC.sort_indices( )
        self.AC.eliminate_zeros( )
        self.AC.sort_indices( )
        
        # Left, row 2
        self.AMD = mAreaCell_phi * vc.mVertex2cell * vc.mDiv_t
        self.AD = mAreaCell_phi * vc.mDiv_v
        self.AMD.eliminate_zeros( )
        self.AMD.sort_indices( )
        self.AD.eliminate_zeros( )
        self.AD.sort_indices( )
        
        # Right, col 2
        self.GN = vc.mGrad_tn * vc.mCell2vertex_n
        self.GN.eliminate_zeros( )
        self.GN.sort_indices( )
        
        # Right, col 1
        self.SN = vc.mSkewgrad_nd * vc.mCell2vertex_psi
        self.SN.eliminate_zeros( )
        self.SN.sort_indices( )

        ## Construct an artificial thickness vector
        thickness_edge = 100 * (10. + np.random.rand(g.nEdges))
        self.thicknessInv = 1. / thickness_edge
#        self.mThicknessInv = eye(g.nEdges)  
#        self.mThicknessInv.data[0,:] = 1./thickness_edge

        # Copy certain matrices over from VectorCalculus; maybe unnecessary
        # in the future.
        self.mSkewgrad_td = vc.mSkewgrad_td.copy( )
        self.mGrad_n_n = vc.mGrad_n_n.copy( )
        
        if c.use_gpu2:
            import cupy as cp
            import cupyx 
            self.AMC = cupyx.scipy.sparse.csr_matrix(self.AMC)
            self.AC = cupyx.scipy.sparse.csr_matrix(self.AC)
            self.AMD = cupyx.scipy.sparse.csr_matrix(self.AMD)
            self.AD= cupyx.scipy.sparse.csr_matrix(self.AD)
            self.GN = cupyx.scipy.sparse.csr_matrix(self.GN)
            self.SN = cupyx.scipy.sparse.csr_matrix(self.SN)
            self.mSkewgrad_td = cupyx.scipy.sparse.csr_matrix(self.mSkewgrad_td)
            self.mGrad_n_n = cupyx.scipy.sparse.csr_matrix(self.mGrad_n_n)
            self.thicknessInv = cp.array(self.thicknessInv)
        
        if c.linear_solver is 'amgx':
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
        
        elif c.linear_solver is 'lu':
            pass
        
        else:
            raise ValueError("Invalid solver choice.")

    def update(self, thickness_edge, vc, c, g):
        if c.use_gpu2:
            self.thicknessInv[:] = cp.array(1./thickness_edge)
        else:
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
        
        if c.linear_solver is 'lu':
            # Convert the matrices to CSC for better performance
            self.A11 = self.A11.tocsc( )
            self.A22 = self.A22.tocsc( )
            
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
            self.d_A11.upload_CSR(self.A11)
            self.d_A22.upload_CSR(self.A22)
            self.slv11.setup(self.d_A11)
            self.slv22.setup(self.d_A22)
        
        else:
            raise ValueError("Invalid solver choice.")


    def solve(self, b1, b2, x, y, env=None, nIter = 10):
        
        if c.linear_solver is 'lu':
            A11_lu = splu(self.A11)
            A22_lu = splu(self.A22)
            b1_new = b1.copy(); b2_new = b2.copy( )
            
            for k in np.arange(nIter):
                b1_new[:] = b1 - self.A12.dot(y)
                b2_new[:] = b2 - self.A21.dot(x)
                x[:] = A11_lu.solve(b1_new)
                y[:] = A22_lu.solve(b2_new)

        elif c.linear_solver is 'amgx':
            if c.use_gpu2:
                b1 = cp.array(b1)
                b2 = cp.array(b2)

            b1_new = b1.copy(); b2_new = b2.copy( )
            self.d_x.upload(x)
            self.d_y.upload(y)
            for k in np.arange(nIter):
                if c.use_gpu2:
                    b1_new[:] = b1 - self.A12.dot(cp.array(y))
                    b2_new[:] = b2 - self.A21.dot(cp.array(x))
                else:
                    b1_new[:] = b1 - self.A12.dot(y)
                    b2_new[:] = b2 - self.A21.dot(x)
                    
                self.d_b1.upload(b1_new)
                self.d_b2.upload(b2_new)
                self.slv11.solve(self.d_b1, self.d_x)
                self.slv22.solve(self.d_b2, self.d_y)
                self.d_x.download(x)
                self.d_y.download(y)
                    

        elif c.linear_solver is 'amg':
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
        
        
