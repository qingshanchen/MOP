import numpy as np
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu, factorized
from swe_comp import swe_comp as cmp
from LinearAlgebra import cg, pcg, cudaCG, cudaPCG
#from pyamg import rootnode_solver
#import time

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
#            if hA.shape[0] in [2562,10242, 40962]:
            if hA.nnz * 1. / hA.shape[0] > 5.5:           # Primary mesh
                AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json'
#            elif hA.shape[0] in [5120, 20480,81920]:
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


            #d_A.upload_CSR(hA)
            d_A.upload(hA.indptr, hA.indices, hA.data)

            # Setup and solve system:
            self.amgx.setup(d_A)

            #self.amgx.solve(b, x)

            #x.download(h_x)
            #print(("rel error for pyamg solver = %e" % (np.sqrt(np.sum((h_x-sol)**2))/np.sqrt(np.sum(sol*sol)))))

            # Clean up:
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
#        print("b max = %e" % np.max(np.abs(b)))
        
        if linear_solver is 'lu':
            x[:] = self.lu.solve(b)
        elif linear_solver is 'cg':
#            print("b max = %e" % np.max(np.abs(b)))
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
            
class VectorCalculus:
    def __init__(self, g, c, env):
        self.env = env

        self.linear_solver = c.linear_solver

        self.max_iter = c.max_iter
        self.err_tol = c.err_tol

        self.areaCell = g.areaCell.copy()
        self.areaTriangle = g.areaTriangle.copy()

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
            D1s_augmented_coo = coo_matrix((valEntries[:nEntries]*self.areaCell[self.cellInterior[rows[:nEntries]]-1], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(nCellsInterior, g.nCells))
            # Convert to csc sparse format
            D1s_augmented = D1s_augmented_coo.tocsc( )

            # Construct a square matrix corresponding to the interior primary cells.
            D1s = D1s_augmented[:, self.cellInterior[:]-1]
            D1s.eliminate_zeros( )

            # Construct the Poisson object
            self.POpd = Poisson(D1s, self.linear_solver, env)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                    g.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*g.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        D2s_coo.eliminate_zeros( )

        self.POpn = Poisson(D2s_coo, self.linear_solver, env)
        

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
            self.POdd = Poisson(E1s_coo, c.linear_solver, env)

        # Construct matrix for discrete Laplacian on the triangles, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, g.verticesOnEdge, \
                      g.dvEdge, g.dcEdge, g.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*g.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        E2s_coo.eliminate_zeros( )
        self.POdn = Poisson(E2s_coo, c.linear_solver, env)

        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mDiv = A.tocsr( )
        self.d_mDiv = Device_CSR(self.mDiv, env)
        
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)


        
