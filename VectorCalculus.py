import numpy as np
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat
from scipy.sparse.linalg import spsolve, splu, factorized
from swe_comp import swe_comp as cmp
from LinearAlgebra import cg, pcg, cudaCG, cudaPCG
#from pyamg import rootnode_solver
#import time

class EllipticCPL:
    def __init__(self, A, linear_solver, env):


        if linear_solver is 'lu':
            self.A = A.tocsc( )
            
        elif linear_solver is 'amgx':
            import pyamgx

            pyamgx.initialize( )

            hA = A.tocsr( )
            AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/FGMRES_AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/AMG_AGGREGATION_CG.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PBICGSTAB_AGGREGATION_W_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json'
 
            cfg = pyamgx.Config( ).create_from_file(AMGX_CONFIG_FILE_NAME) 
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

        self.max_iter = c.max_iter
        self.err_tol = c.err_tol

        self.areaCell = g.areaCell.copy()
        self.areaTriangle = g.areaTriangle.copy()

        self.on_a_global_sphere = c.on_a_global_sphere

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

            # Construct matrix for discrete Laplacian on interior cells, with homogeneous Dirichlet BC
            nEntries, rows, cols, valEntries = \
              cmp.construct_discrete_laplace_interior(g.boundaryEdgeMark[:], \
                        g.cellsOnEdge, g.boundaryCellMark, \
                        self.cellRankInterior, g.dvEdge, g.dcEdge, \
                        g.areaCell)
            D1s_augmented_coo = coo_matrix(( \
                        valEntries[:nEntries]*self.areaCell[self.cellInterior[rows[:nEntries]]-1], \
                        (rows[:nEntries], cols[:nEntries])), shape=(nCellsInterior, g.nCells))
            # Convert to csc sparse format
            D1s_augmented = D1s_augmented_coo.tocsc( )

            # Construct a square matrix corresponding to the interior primary cells.
            D1s = D1s_augmented[:, self.cellInterior[:]-1]
            D1s.eliminate_zeros( )

            # Construct the Poisson object
#            self.POpd = Poisson(D1s, self.linear_solver, env)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                    g.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*g.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        D2s_coo.eliminate_zeros( )

        #self.POpn = Poisson(D2s_coo, self.linear_solver, env)
        

        if not c.on_a_global_sphere:
            # Construct matrix for discrete Laplacian on the triangles, corresponding to
            # the homogeneous Dirichlet boundary conditions
            nEntries, rows, cols, valEntries = \
              cmp.construct_discrete_laplace_triangle(g.boundaryEdgeMark[:], \
                            g.verticesOnEdge, g.dvEdge, g.dcEdge, g.areaTriangle)
            E1_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(g.nVertices, g.nVertices))
            E1s_coo = coo_matrix((valEntries[:nEntries]*self.areaTriangle[rows[:nEntries]], \
                                  (rows[:nEntries], cols[:nEntries])), shape=(g.nVertices, g.nVertices))
            E1s_coo.eliminate_zeros( )
#            self.POdd = Poisson(E1s_coo, c.linear_solver, env)

        # Construct matrix for discrete Laplacian on the triangles, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, g.verticesOnEdge, \
                      g.dvEdge, g.dcEdge, g.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*g.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        E2s_coo.eliminate_zeros( )
#        self.POdn = Poisson(E2s_coo, c.linear_solver, env)

        ## Construct the matrix representing the discrete div (on primal mesh)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mDiv = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv = Device_CSR(self.mDiv, env)

        ## Construct the matrix representing the discrete curl (on primal mesh)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mCurl = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl = Device_CSR(self.mCurl, env)

        ## Construct the matrix representing the discrete curl (on dual mesh)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mCurl_trig = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_trig = Device_CSR(self.mCurl_trig, env)
            
        ## Construct the matrix representing the discrete Laplace operator (primal
        ## mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, \
                                                  g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        self.mLaplace = A.tocsr( )

        if c.use_gpu:
            self.d_mLaplace = Device_CSR(self.mLaplace, env)

        ## Construct the matrix representing the discrete grad operator for the primal
        ## mesh. 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mGradn = A.tocsr( )

        if c.use_gpu:
            self.d_mGradn = Device_CSR(self.mGradn, env)

        ## Construct the matrix representing the discrete grad operator for the primal
        ## mesh. phi is assume to be zero at index = 0.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_gradn_phi(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mGradn_phi = A.tocsr( )

        if c.use_gpu:
            self.d_mGradn_phi = Device_CSR(self.mGradn_phi, env)
            
        ## Construct the matrix representing the discrete skew grad operator 
        ## on the dual mesh; homogeneous Dirichlet assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mSkewgrad = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad = Device_CSR(self.mSkewgrad, env)
            

        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Dirichlet BC's 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_td(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_td = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_td = Device_CSR(self.mGrad_td, env)

        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Neumann BC's
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_tn(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_tn = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_tn = Device_CSR(self.mGrad_tn, env)


        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex = Device_CSR(self.mCell2vertex, env)

        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh; homogeneous Dirichlet BC's are assumed
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex_psi(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.boundaryCellMark, c.on_a_global_sphere)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex_psi = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex_psi = Device_CSR(self.mCell2vertex_psi, env)
            
        ## Construct the matrix representing the mapping from the dual mesh onto the primal
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_vertex2cell(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nVertices))
        self.mVertex2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mVertex2cell = Device_CSR(self.mVertex2cell, env)

        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2edge = Device_CSR(self.mCell2edge, env)

        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mEdge2cell = Device_CSR(self.mEdge2cell, env)


        # A diagonal matrix representing scaling by cell areas
        self.mAreaCell = diags(g.areaCell, 0, format='csr')
        self.mAreaCell_phi = self.mAreaCell.copy( )
        self.mAreaCell_phi[0,0] = 0.
        self.mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            self.mAreaCell_psi = self.mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[self.cellBoundary - 1] = 0.
            self.mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            self.mAreaCell_psi.eliminate_zeros( )
            
        if c.use_gpu:                        # Need to update at every step
            self.d_mAreaCell = Device_CSR(self.mAreaCell, env)

        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi
        AMC = self.mAreaCell_psi * self.mVertex2cell * self.mCurl_trig
        AD = self.mAreaCell_phi * self.mDiv
        SN = self.mSkewgrad * self.mCell2vertex_psi
        
        self.leftM = bmat([[AMC],[AD]], format='csr')
        self.rightM = bmat([[SN, self.mGradn_phi]], format='csr')

        self.leftM.eliminate_zeros( )
        self.rightM.eliminate_zeros( )

        self.mThicknessInv = eye(g.nEdges)   # This is only a space holder
#        if c.use_gpu:                        # Need to update at every step
#            d_mThicknessInv = Device_CSR(self.mThicknessInv.to_csr(), env)
        
        thickness_edge = np.zeros(g.nEdges)
        thickness_edge[:] = 1000.    # Any non-zero should suffice
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        self.coefM = self.leftM * self.mThicknessInv * self.rightM

        if c.on_a_global_sphere:
            self.coefM[0,0] = 1.
            self.coefM[g.nCells, g.nCells] = 1.
        else:
            self.coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            self.coefM[g.nCells, g.nCells] = 1.
        
        self.POcpl = EllipticCPL(self.coefM, c.linear_solver, env)
            
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)

            
    def discrete_div(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mDiv.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mDiv.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mDiv.shape[0], \
                n=self.d_mDiv.shape[1], nnz=self.d_mDiv.nnz, alpha=1.0, \
                descr=self.d_mDiv.cuSparseDescr, csrVal=self.d_mDiv.dData, \
                csrRowPtr=self.d_mDiv.dPtr, csrColInd=self.d_mDiv.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mDiv.dot(vEdge)


    def discrete_curl(self, vEdge):
        '''
        The discrete curl operator on the primal mesh.
        No-slip boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl.shape[0], \
                n=self.d_mCurl.shape[1], nnz=self.d_mCurl.nnz, alpha=1.0, \
                descr=self.d_mCurl.cuSparseDescr, csrVal=self.d_mCurl.dData, \
                csrRowPtr=self.d_mCurl.dPtr, csrColInd=self.d_mCurl.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl.dot(vEdge)


    def discrete_curl_trig(self, vEdge):
        '''
        The discrete curl operator on the dual mesh.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl_trig.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl_trig.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl_trig.shape[0], \
                n=self.d_mCurl_trig.shape[1], nnz=self.d_mCurl_trig.nnz, alpha=1.0, \
                descr=self.d_mCurl_trig.cuSparseDescr, csrVal=self.d_mCurl_trig.dData, \
                csrRowPtr=self.d_mCurl_trig.dPtr, csrColInd=self.d_mCurl_trig.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl_trig.dot(vEdge)
        

    def discrete_laplace(self, sCell):
        '''
        No-slip boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(sCell) == self.d_mLaplace.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mLaplace.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mLaplace.shape[0], \
                n=self.d_mLaplace.shape[1], nnz=self.d_mLaplace.nnz, alpha=1.0, \
                descr=self.d_mLaplace.cuSparseDescr, csrVal=self.d_mLaplace.dData, \
                csrRowPtr=self.d_mLaplace.dPtr, csrColInd=self.d_mLaplace.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mLaplace.dot(sCell)


    # The discrete gradient operator on the primal mesh
    def discrete_grad_n(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mGradn.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mGradn.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGradn.shape[0], \
                n=self.d_mGradn.shape[1], nnz=self.d_mGradn.nnz, alpha=1.0, \
                descr=self.d_mGradn.cuSparseDescr, csrVal=self.d_mGradn.dData, \
                csrRowPtr=self.d_mGradn.dPtr, csrColInd=self.d_mGradn.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGradn.dot(sCell)


    # The discrete gradient operator on the dual mesh, assuming
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


    # The discrete gradient operator on the dual mesh, assuming
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


    def update_matrix_for_coupled_elliptic(self, thickness_edge, c, g):
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        self.coefM = self.leftM * self.mThicknessInv * self.rightM

        if c.on_a_global_sphere:
            self.coefM[0,0] = 1.
            self.coefM[g.nCells, g.nCells] = 1.
        else:
            self.coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            self.coefM[g.nCells, g.nCells] = 1.
        
        
        
        

        
