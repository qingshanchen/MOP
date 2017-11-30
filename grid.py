import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu
import netCDF4 as nc
from swe_comp import swe_comp as cmp
import os

max_int = np.iinfo('int32').max

class grid_data:
    
    def __init__(self, netcdf_file, c):
        # Read the grid file
        grid = nc.Dataset(netcdf_file,'r')
    
        # Read relative dimensions
        self.nCells = len(grid.dimensions['nCells'])
        self.nEdges = len(grid.dimensions['nEdges'])
        self.nVertices = len(grid.dimensions['nVertices'])
        self.vertexDegree = len(grid.dimensions['vertexDegree'])
        self.nVertLevels = len(grid.dimensions['nVertLevels'])

        # Read grid informaton
        xCell = grid.variables['xCell'][:]
        yCell = grid.variables['yCell'][:]
        zCell = grid.variables['zCell'][:]
        xEdge = grid.variables['xEdge'][:]
        yEdge = grid.variables['yEdge'][:]
        zEdge = grid.variables['zEdge'][:]
        xVertex = grid.variables['xVertex'][:]
        yVertex = grid.variables['yVertex'][:]
        zVertex = grid.variables['zVertex'][:]
        self.latCell = grid.variables['latCell'][:]
        self.lonCell = grid.variables['lonCell'][:]
        self.latEdge = grid.variables['latEdge'][:]
        self.lonEdge = grid.variables['lonEdge'][:]
        self.latVertex = grid.variables['latVertex'][:]
        self.lonVertex = grid.variables['lonVertex'][:]
        self.cellsOnEdge = grid.variables['cellsOnEdge'][:]
        self.nEdgesOnCell = grid.variables['nEdgesOnCell'][:]
        self.dvEdge = grid.variables['dvEdge'][:]
        self.dcEdge = grid.variables['dcEdge'][:]
        self.areaCell = grid.variables['areaCell'][:]
        self.areaTriangle = grid.variables['areaTriangle'][:]
        self.cellsOnCell = grid.variables['cellsOnCell'][:]
        self.verticesOnCell = grid.variables['verticesOnCell'][:]
        self.edgesOnCell = grid.variables['edgesOnCell'][:]
        self.verticesOnEdge = grid.variables['verticesOnEdge'][:]
        self.edgesOnVertex = grid.variables['edgesOnVertex'][:]
        self.cellsOnVertex = grid.variables['cellsOnVertex'][:]
        self.kiteAreasOnVertex = grid.variables['kiteAreasOnVertex'][:]

        self.fEdge = grid.variables['fEdge'][:]
        self.fVertex = grid.variables['fVertex'][:]
        self.fCell = 2 * 7.292e-5 * np.sin(self.latCell[:])
        #self.fCell = grid.variables['fCell'][:]


        if c.on_a_global_sphere:
            self.boundaryEdgeMark = np.zeros(self.nEdges).astype('int32')
            self.boundaryEdgeMark[:] = 0
            self.boundaryCellMark = np.zeros(self.nCells).astype('int32')
            self.boundaryCellMark[:] = 0
        else:
            self.boundaryEdgeMark = grid.variables['boundaryEdgeMark'][:]
            self.boundaryCellMark = grid.variables['boundaryCellMark'][:] 

        radius = np.sqrt(xCell**2 + yCell**2 + zCell**2)
        if np.max(np.abs(radius - 1.)/1.) < 0.01:
            # To scale the coordinates
            xCell *= c.earth_radius
            yCell *= c.earth_radius
            zCell *= c.earth_radius
            xEdge *= c.earth_radius
            yEdge *= c.earth_radius
            zEdge *= c.earth_radius
            xVertex *= c.earth_radius
            yVertex *= c.earth_radius
            zVertex *= c.earth_radius
            self.dvEdge *= c.earth_radius
            self.dcEdge *= c.earth_radius
            self.areaCell *= c.earth_radius**2
            self.areaTriangle *= c.earth_radius**2
            self.kiteAreasOnVertex *= c.earth_radius**2
        elif np.max(np.abs(radius-c.earth_radius)/c.earth_radius) < 0.01:
            pass
        else:
            raise ValueError("Unknown domain raius.")
            
        # Create new grid_data variables
        self.bottomTopographyCell = np.zeros(self.nCells)
        self.bottomTopographyVertex = np.zeros(self.nVertices)

        if c.use_direct_solver:
            self.solver_choice = 'direct'
        else:
            self.solver_choice = 'cg'

        grid.close()

        if not c.on_a_global_sphere:
            # Collect non-boundary (interior) cells and put into a vector,
            # and boundary cells into a separate vector
            nCellsBoundary = np.sum(self.boundaryCellMark[:]>0)
            nCellsInterior = self.nCells - nCellsBoundary
            self.cellInterior, self.cellBoundary, self.cellRankInterior, \
                cellInner_tmp, cellOuter_tmp, self.cellRankInner, \
                nCellsInner, nCellsOuter = \
                cmp.separate_boundary_interior_inner_cells(nCellsInterior,  \
                nCellsBoundary, max_int, self.boundaryCellMark, self.cellsOnCell, self.nEdgesOnCell)
            self.cellInner = cellInner_tmp[:nCellsInner]
            self.cellOuter = cellOuter_tmp[:nCellsOuter]

            # Construct matrix for discrete Laplacian on interior cells, with homogeneous Dirichlet BC
            nEntries, rows, cols, valEntries = \
              cmp.construct_discrete_laplace_interior(self.boundaryEdgeMark[:], \
                        self.cellsOnEdge, self.boundaryCellMark, \
                        self.cellRankInterior, self.dvEdge, self.dcEdge, \
                        self.areaCell)
            D1_augmented_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(nCellsInterior, self.nCells))
            # Convert to csc sparse format
            self.D1_augmented = D1_augmented_coo.tocsc( )

            # Construct a square matrix corresponding to the interior primary cells.
            self.D1 = self.D1_augmented[:, self.cellInterior[:]-1]
            self.D1_bdry = self.D1_augmented[:, self.cellBoundary[:]-1]

            self.lu_D1 = splu(self.D1)

        # Construct matrix for discrete Laplacian on all cells, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(self.cellsOnEdge, self.dvEdge, self.dcEdge, \
                    self.areaCell)
        D2s_coo = coo_matrix((valEntries[:nEntries]*self.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(self.nCells, self.nCells))
        
        # Convert to csc sparse format

        if c.use_direct_solver:
            self.D2s = D2s_coo.tocsc( )
            self.lu_D2s = splu(self.D2s)
        else:
            self.D2s = D2s_coo.tocsr( )
#            self.D2spd = -self.D2s
#            B = np.ones((self.D2spd.shape[0],1), dtype=self.D2spd.dtype); BH = B.copy()
#            self.D2spd_amg = rootnode_solver(self.D2spd, B=B, BH=BH,
#                strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
#                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
#                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                aggregate="standard",
#                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
#                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
#                max_levels=15,
#                max_coarse=300,
#                coarse_solver="pinv")

        if not c.on_a_global_sphere:
            # Construct matrix for discrete Laplacian on the triangles, corresponding to
            # the homogeneous Dirichlet boundary conditions
            nEntries, rows, cols, valEntries = \
              cmp.construct_discrete_laplace_triangle(self.boundaryEdgeMark[:], \
                            self.verticesOnEdge, self.dvEdge, self.dcEdge, self.areaTriangle)
            E1_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                                   cols[:nEntries])), shape=(self.nVertices, self.nVertices))
            # Convert to csc sparse format
            self.E1 = E1_coo.tocsc( )

            self.lu_E1 = splu(self.E1)

        # Construct matrix for discrete Laplacian on the triangles, corresponding to the
        # Poisson problem with Neumann BC's, or to the Poisson problem on a global sphere (no boundary)
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(self.boundaryEdgeMark, self.verticesOnEdge, \
                      self.dvEdge, self.dcEdge, self.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*self.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(self.nVertices, self.nVertices))
        # Convert to csc sparse format

        if c.use_direct_solver:
            self.E2s = E2s_coo.tocsc( )
            self.lu_E2s = splu(self.E2s)
        else:
            self.E2s = E2s_coo.tocsr( )
#            self.E2spd = -self.E2s
#            B = np.ones((self.E2spd.shape[0],1), dtype=self.E2spd.dtype); BH = B.copy()
#            self.E2spd_amg = rootnode_solver(self.E2spd, B=B, BH=BH,
#                strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
#                smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
#                improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
#                aggregate="standard",
#                presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
#                postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
#                max_levels=15,
#                max_coarse=300,
#                coarse_solver="pinv")

        # Make a copy of grid file
        os.system('cp %s %s' % (netcdf_file, c.output_file))

        # Open the output file to save scaled grid data
        out = nc.Dataset(c.output_file, 'a', format='NETCDF4_CLASSIC')
        out.variables['xCell'][:] = xCell[:]
        out.variables['yCell'][:] = yCell[:]
        out.variables['zCell'][:] = zCell[:]
        out.variables['xEdge'][:] = xEdge[:]
        out.variables['yEdge'][:] = yEdge[:]
        out.variables['zEdge'][:] = zEdge[:]
        out.variables['xVertex'][:] = xVertex[:]
        out.variables['yVertex'][:] = yVertex[:]
        out.variables['zVertex'][:] = zVertex[:]
        out.variables['dvEdge'][:] = self.dvEdge[:]
        out.variables['dcEdge'][:] = self.dcEdge[:]
        out.variables['areaCell'][:] = self.areaCell[:]
        out.variables['areaTriangle'][:] = self.areaTriangle[:]
        out.variables['kiteAreasOnVertex'][:] = self.kiteAreasOnVertex[:]

        out.close( )

    def invLaplace_prime_dirich(self, b, x):

        if self.solver_choice is 'direct':
            x[:] = 0.    # Homogeneous Dirichlet BC
            x[self.cellInterior[:]-1] = self.lu_D1.solve(b[self.cellInterior[:]-1])
        else:
            raise ValueError("Indicator for solver is not valid. Abort.")
            
            
            
        
