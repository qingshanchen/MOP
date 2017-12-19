import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu, factorized
from swe_comp import swe_comp as cmp
from LinearAlgebra import cg, pcg, cudaCG
#from pyamg import rootnode_solver
from accelerate import cuda
cuSparse = cuda.sparse.Sparse( )
from numba import cuda
import time

class Poisson:
    def __init__(self, A):
        self.A = A.copy( )
        self.dData = cuda.to_device(self.A.data)
        self.dPtr = cuda.to_device(self.A.indptr)
        self.dInd = cuda.to_device(self.A.indices)
        self.cuSparseDescr = cuSparse.matdescr( )


class PoissonSPD:
    def __init__(self, A):
        self.A = A.copy( )
        self.Adata = cuda.to_device(self.A.data)
        self.Aptr = cuda.to_device(self.A.indptr)
        self.Aind = cuda.to_device(self.A.indices)
        self.Adescr = cuSparse.matdescr( )
        
        A_t = self.A.copy( )
        A_t.data = np.where(A_t.nonzero()[0] >= A_t.nonzero()[1], A_t.data, 0.)
        A_t.eliminate_zeros( )
        A_t_descr = cuSparse.matdescr(matrixtype='S', fillmode='L')
        info = cuSparse.csrsv_analysis(trans='N', m=A_t.shape[0], nnz=A_t.nnz, \
                                   descr=A_t_descr, csrVal=A_t.data, \
                                   csrRowPtr=A_t.indptr, csrColInd=A_t.indices)
        cuSparse.csric0(trans='N', m=A_t.shape[0], \
                    descr=A_t_descr, csrValM=A_t.data, csrRowPtrA=A_t.indptr,\
                    csrColIndA=A_t.indices, info=info)

        self.L = A_t
        self.Lmv_descr = cuSparse.matdescr( )
        self.Lsv_descr = cuSparse.matdescr(matrixtype='T', fillmode='L')
        self.Ldata = cuda.to_device(self.L.data)
        self.Lptr = cuda.to_device(self.L.indptr)
        self.Lind = cuda.to_device(self.L.indices)
        self.Lsv_info = cuSparse.csrsv_analysis(trans='N', m=self.L.shape[0], \
                nnz=self.L.nnz,  descr=self.Lsv_descr, csrVal=self.Ldata, \
                csrRowPtr=self.Lptr, csrColInd=self.Lind)        
        
        
        self.LT = self.L.transpose( )
        self.LT.tocsr( )
        self.LTmv_descr = cuSparse.matdescr( )
        self.LTsv_descr = cuSparse.matdescr(matrixtype='T', fillmode='U')
        self.LTdata = cuda.to_device(self.LT.data)
        self.LTptr = cuda.to_device(self.LT.indptr)
        self.LTind = cuda.to_device(self.LT.indices)
        self.LTsv_info = cuSparse.csrsv_analysis(trans='N', m=self.LT.shape[0], \
                 nnz=self.LT.nnz, descr=self.LTsv_descr, csrVal=self.LTdata, \
                 csrRowPtr=self.LTptr, csrColInd=self.LTind)        
        

            
class VectorCalculus:
    def __init__(self, g, c):

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
            D1_augmented_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(nCellsInterior, g.nCells))
            D1s_augmented_coo = coo_matrix((valEntries[:nEntries]*self.areaCell[self.cellInterior[rows[:nEntries]]-1], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(nCellsInterior, g.nCells))
            # Convert to csc sparse format
            self.D1_augmented = D1_augmented_coo.tocsc( )
            self.D1s_augmented = D1s_augmented_coo.tocsc( )

            # Construct a square matrix corresponding to the interior primary cells.
            self.D1 = self.D1_augmented[:, self.cellInterior[:]-1]
            self.D1_bdry = self.D1_augmented[:, self.cellBoundary[:]-1]
            self.D1s = self.D1s_augmented[:, self.cellInterior[:]-1]

            self.lu_D1 = splu(self.D1)
            self.lu_D1s = splu(self.D1s)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                    g.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*g.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        D2s_coo.eliminate_zeros( )
        
        # Convert to csc sparse format

        if self.linear_solver is 'lu':
            self.D2s = D2s_coo.tocsc( )
            self.lu_D2s = splu(self.D2s)
        elif self.linear_solver is 'cg':
            self.D2s = D2s_coo.tocsr( )
            self.POpn = Poisson(self.D2s)
            self.POpnSPD = PoissonSPD(-self.D2s)
        elif self.linear_solver is 'pcg':
            self.D2s = D2s_coo.tocsr( )
            self.D2s = -self.D2s
            D2s_t = self.D2s.copy( )
            D2s_t.data = np.where(D2s_t.nonzero()[0] >= D2s_t.nonzero()[1], D2s_t.data, 0.)
            D2s_t.eliminate_zeros( )
            D2s_t_descr = cuSparse.matdescr(matrixtype='S', fillmode='L')
            info = cuSparse.csrsv_analysis(trans='N', m=D2s_t.shape[0], nnz=D2s_t.nnz, \
                                       descr=D2s_t_descr, csrVal=D2s_t.data, \
                                       csrRowPtr=D2s_t.indptr, csrColInd=D2s_t.indices)
            cuSparse.csric0(trans='N', m=D2s_t.shape[0], \
                        descr=D2s_descr, csrValM=D2s_t.data, csrRowPtrA=D2s_t.indptr,\
                        csrColIndA=D2s_t.indices, info=info)
            self.D2sL = D2s_t
            self.D2sL_solve = factorized(self.D2sL)
            self.D2sLT = self.D2sL.transpose( )
            self.D2sLT.tocsr( )
            self.D2sLT_solve = factorized(self.D2sLT)

            
        elif self.linear_solver is 'amg':
            D2s = D2s_coo.tocsr( )
            D2s = -D2s
            B = np.ones((D2s.shape[0],1), dtype=D2s.dtype); BH = B.copy()
            self.D2spd_amg = rootnode_solver(D2s, B=B, BH=BH,
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
        else:
            raise ValueError("Invalid solver choice.")

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
                                   
            # Convert to csc sparse format
            self.E1 = E1_coo.tocsc( )
            self.E1s = E1s_coo.tocsc( )

            self.lu_E1 = splu(self.E1)
            self.lu_E1s = splu(self.E1s)

        # Construct matrix for discrete Laplacian on the triangles, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, g.verticesOnEdge, \
                      g.dvEdge, g.dcEdge, g.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*g.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        # Convert to csc sparse format

        if self.linear_solver is 'lu':
            self.E2s = E2s_coo.tocsc( )
            self.lu_E2s = splu(self.E2s)
        elif self.linear_solver is 'cg':
            self.E2s = E2s_coo.tocsr( )
        elif self.linear_solver is 'amg':
            E2s = E2s_coo.tocsr( )
            E2s = -E2s
            B = np.ones((E2s.shape[0],1), dtype=E2s.dtype); BH = B.copy()
            self.E2spd_amg = rootnode_solver(E2s, B=B, BH=BH,
                strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), \
                                    None, None, None, None, None, None, None, None, None, None, None, \
                                    None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")
        else:
            raise ValueError("Invalid solver choice!")
        

        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)


        
    def invLaplace_prime_dirich(self, b, x):

        self.scalar_cell[:] = b*self.areaCell
        
        if self.linear_solver is 'lu':
            x[self.cellInterior[:]-1] = self.lu_D1s.solve(self.scalar_cell[self.cellInterior[:]-1])
            x[self.cellBoundary[:]-1] = 0.              # Re-enforce the Dirichlet BC
        
        elif self.linear_solver is 'cg':
            self.scalar_cell_interior[:] = x[self.cellInterior[:]-1].copy( )   # Copy over initial guesses
            info, nIter = cg(self.D1s, self.scalar_cell[self.cellInterior[:]-1], \
                             self.scalar_cell_interior, max_iter=self.max_iter, relres = self.err_tol)
            x[self.cellInterior[:]-1] = self.scalar_cell_interior[:]
            x[self.cellBoundary[:]-1] = 0.              # Re-enforce the Dirichlet BC

        else:
            raise ValueError("Invalid solver choice.")


    def invLaplace_prime_neumann(self, b, x):
        self.scalar_cell[:] = self.areaCell * b
        self.scalar_cell[0] = 0.    # Set to zero to make x[0] zero

        if self.linear_solver is 'lu':
            x[:] = self.lu_D2s.solve(self.scalar_cell)

        elif self.linear_solver is 'cg':
            x[:] -= x[0]
            
            t0 = time.time( )
#            info, nIter = cg(self.D2s, self.scalar_cell, x, max_iter=self.max_iter, relres = self.err_tol)
            info, nIter = cudaCG(self.POpn, self.scalar_cell, x, max_iter=self.max_iter, relres = self.err_tol)
            t1 = time.time( )
            print("D2s, nIter & walltime = %d, %f" % (nIter,t1-t0))
#            print("Wall time: %f " % (t1-t0))

        elif self.linear_solver is 'pcg':
            x[:] -= x[0]
            self.scalar_cell = -self.scalar_cell
            info, nIter = pcg(self.D2s, self.D2sL, self.D2sLT, self.scalar_cell, x, max_iter=self.max_iter, relres = self.err_tol)
            print("D2s, nIter = %d" % nIter)

        else:
            raise ValueError("Indicator for solver is not valid. Abort.")

        #### For AMG solver:
                #res = []
                #b = -b
                #self.psi_cell[:] = g.D2spd_amg.solve(b, x0=self.psi_cell, tol=c.err_tol, residuals=res, accel="cg", maxiter=300, cycle="V")
                #print("compute_psi_cell, nIter = %d" % len(res))
                #print(res)

    def invLaplace_dual_dirich(self, b, x):

        self.scalar_vertex[:] = b * self.areaTriangle
        
        if self.linear_solver is 'lu':
            x[:] = self.lu_E1s.solve(self.scalar_vertex)
            
        elif self.linear_solver is 'cg':
            info, nIter = cg(self.E1s, self.scalar_vertex, \
                             x, max_iter=self.max_iter, relres = self.err_tol)
            
        else:
            raise ValueError("Indicator for solver is not valid. Abort.")
        

    def invLaplace_dual_neumann(self, b, x):
        self.scalar_vertex[:] = self.areaTriangle * b   #Scaling
        self.scalar_vertex[0] = 0.   #Set to zero to make x[0] zero

        if self.linear_solver is 'lu':
            x[:] = self.lu_E2s.solve(self.scalar_vertex)
        elif self.linear_solver is 'cg':
            x[:] -= x[0]
            info, nIter = cg(self.E2s, self.scalar_vertex, x, max_iter=self.max_iter, relres = self.err_tol)

        else:
            raise ValueError("Invalid solver choice.")
        
