import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu
from swe_comp import swe_comp as cmp

class VectorCalculus:
    def __init__(self, g, c):
        if c.use_direct_solver:
            self.solver_choice = 'direct'
        else:
            self.solver_choice = 'cg'

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
                        g.cellRankInterior, g.dvEdge, g.dcEdge, \
                        g.areaCell)
            D1_augmented_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(nCellsInterior, g.nCells))
            # Convert to csc sparse format
            self.D1_augmented = D1_augmented_coo.tocsc( )

            # Construct a square matrix corresponding to the interior primary cells.
            self.D1 = self.D1_augmented[:, g.cellInterior[:]-1]
            self.D1_bdry = self.D1_augmented[:, g.cellBoundary[:]-1]

            self.lu_D1 = splu(self.D1)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(g.cellsOnEdge, g.dvEdge, g.dcEdge, \
                    g.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*g.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        
        # Convert to csc sparse format

        if c.use_direct_solver:
            self.D2s = D2s_coo.tocsc( )
            self.lu_D2s = splu(self.D2s)
        else:
            self.D2s = D2s_coo.tocsr( )
            self.D2spd = -self.D2s
            B = np.ones((self.D2spd.shape[0],1), dtype=self.D2spd.dtype); BH = B.copy()
            self.D2spd_amg = rootnode_solver(self.D2spd, B=B, BH=BH,
                strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")

        if not c.on_a_global_sphere:
            # Construct matrix for discrete Laplacian on the triangles, corresponding to
            # the homogeneous Dirichlet boundary conditions
            nEntries, rows, cols, valEntries = \
              cmp.construct_discrete_laplace_triangle(g.boundaryEdgeMark[:], \
                            g.verticesOnEdge, g.dvEdge, g.dcEdge, g.areaTriangle)
            E1_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(g.nVertices, g.nVertices))
            # Convert to csc sparse format
            self.E1 = E1_coo.tocsc( )

            self.lu_E1 = splu(self.E1)

        # Construct matrix for discrete Laplacian on the triangles, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, g.verticesOnEdge, \
                      g.dvEdge, g.dcEdge, g.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*g.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        # Convert to csc sparse format

        if c.use_direct_solver:
            self.E2s = E2s_coo.tocsc( )
            self.lu_E2s = splu(self.E2s)
        else:
            self.E2s = E2s_coo.tocsr( )
            self.E2spd = -self.E2s
            B = np.ones((self.E2spd.shape[0],1), dtype=self.E2spd.dtype); BH = B.copy()
            self.E2spd_amg = rootnode_solver(self.E2spd, B=B, BH=BH,
                strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                aggregate="standard",
                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver="pinv")
        

    def invLaplace_prime_dirich(self, b, x):

        if self.solver_choice is 'direct':
            x[:] = 0.    # Homogeneous Dirichlet BC
            x[self.cellInterior[:]-1] = self.lu_D1.solve(b[self.cellInterior[:]-1])
        else:
            raise ValueError("Indicator for solver is not valid. Abort.")
            
