import numpy as np
#from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from swe_comp import swe_comp as cmp
import netCDF4 as nc
import os

max_int = np.iinfo('int32').max

class grid_data:
    
    def __init__(self, netcdf_file, c):
        # load appropriate module for defining arrays on GPU or CPU
        if c.use_gpu:
            import cupy as xp
        else:
            import numpy as xp

        # Read the grid file
        grid = nc.Dataset(netcdf_file,'r')
    
        # Read relative dimensions
        self.nCells = len(grid.dimensions['nCells'])
        self.nEdges = len(grid.dimensions['nEdges'])
        self.nVertices = len(grid.dimensions['nVertices'])
        self.vertexDegree = len(grid.dimensions['vertexDegree'])

        #
        # Read grid informaton
        #

        # Local variables that are only read and written here
        xCell = grid.variables['xCell'][:]
        yCell = grid.variables['yCell'][:]
        zCell = grid.variables['zCell'][:]
        xEdge = grid.variables['xEdge'][:]
        yEdge = grid.variables['yEdge'][:]
        zEdge = grid.variables['zEdge'][:]
        xVertex = grid.variables['xVertex'][:]
        yVertex = grid.variables['yVertex'][:]
        zVertex = grid.variables['zVertex'][:]

        # Variables that are needed by the Fortran routines only
        self.dvEdge = grid.variables['dvEdge'][:]
        self.dcEdge = grid.variables['dcEdge'][:]
        self.cellsOnEdge = grid.variables['cellsOnEdge'][:]
        self.nEdgesOnCell = grid.variables['nEdgesOnCell'][:]
        self.cellsOnCell = grid.variables['cellsOnCell'][:]
        self.verticesOnCell = grid.variables['verticesOnCell'][:]
        self.edgesOnCell = grid.variables['edgesOnCell'][:]
        self.verticesOnEdge = grid.variables['verticesOnEdge'][:]
        self.edgesOnVertex = grid.variables['edgesOnVertex'][:]
        self.cellsOnVertex = grid.variables['cellsOnVertex'][:]
        self.kiteAreasOnVertex = grid.variables['kiteAreasOnVertex'][:]
        self.boundaryEdgeMark = grid.variables['boundaryEdgeMark'][:]
        self.boundaryCellMark = grid.variables['boundaryCellMark'][:]

        # Variables that are lightly used in a simulation run
        self.latCell = xp.asarray(grid.variables['latCell'][:])
        self.lonCell = xp.asarray(grid.variables['lonCell'][:])
        self.latEdge = xp.asarray(grid.variables['latEdge'][:])
        self.lonEdge = xp.asarray(grid.variables['lonEdge'][:])
        self.latVertex = xp.asarray(grid.variables['latVertex'][:])
        self.lonVertex = xp.asarray(grid.variables['lonVertex'][:])

        # Variables regularly used in a simulation run
        self.areaCell = xp.asarray(grid.variables['areaCell'][:])
        self.areaTriangle = xp.asarray(grid.variables['areaTriangle'][:])
        self.fEdge = xp.asarray(grid.variables['fEdge'][:])
        self.fVertex = xp.asarray(grid.variables['fVertex'][:])
        self.fCell = 2 * 7.292e-5 * xp.sin(self.latCell[:])
        #self.fCell = grid.variables['fCell'][:]

        # To decide whether the domain is on a sphere
        rad2 = xCell**2 + yCell**2 + zCell**2
        rad_mean = np.sqrt(np.mean(rad2))
        mean_dev = np.sqrt(np.mean((np.sqrt(rad2) - rad_mean)**2))
        if mean_dev / rad_mean < 0.1:
            c.on_a_sphere = True
        
        if np.sum(self.boundaryCellMark) == 0:
            c.on_a_global_sphere = True
        else:
            c.on_a_global_sphere = False


        radius = np.sqrt(xCell**2 + yCell**2 + zCell**2)
        if np.max(np.abs(radius - 1.)/1.) < 0.01:
            # To scale the coordinates
            xCell *= c.sphere_radius
            yCell *= c.sphere_radius
            zCell *= c.sphere_radius
            xEdge *= c.sphere_radius
            yEdge *= c.sphere_radius
            zEdge *= c.sphere_radius
            xVertex *= c.sphere_radius
            yVertex *= c.sphere_radius
            zVertex *= c.sphere_radius
            self.dvEdge *= c.sphere_radius
            self.dcEdge *= c.sphere_radius
            self.areaCell *= c.sphere_radius**2
            self.areaTriangle *= c.sphere_radius**2
            self.kiteAreasOnVertex *= c.sphere_radius**2
        elif np.max(np.abs(radius-c.sphere_radius)/c.sphere_radius) < 0.01:
            pass
        else:
            raise ValueError("Unknown domain raius.")
            
        # Create new grid_data variables
        self.bottomTopographyCell = xp.zeros(self.nCells)
        self.bottomTopographyVertex = xp.zeros(self.nVertices)
        self.areaEdge = xp.asarray(self.dvEdge * self.dcEdge / 2.)

        grid.close()

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
        if c.use_gpu:
            out.variables['areaCell'][:] = self.areaCell.get()
            out.variables['areaTriangle'][:] = self.areaTriangle.get()
        else:        
            out.variables['areaCell'][:] = self.areaCell[:]
            out.variables['areaTriangle'][:] = self.areaTriangle[:]
        out.variables['kiteAreasOnVertex'][:] = self.kiteAreasOnVertex[:]

        out.close( )

        #
        # Computed mesh element indices
        #
        if not c.on_a_global_sphere:
            # Collect non-boundary (interior) cells and put into a vector,
            # and boundary cells into a separate vector
            nCellsBoundary = np.sum(self.boundaryCellMark[:]>0)
            nCellsInterior = self.nCells - nCellsBoundary
            
            self.cellInterior, self.cellBoundary, self.cellRankInterior, \
                cellInner_tmp, cellOuter_tmp, self.cellRankInner, \
                nCellsInner, nCellsOuter = \
                cmp.separate_boundary_interior_inner_cells(nCellsInterior,  \
                nCellsBoundary, c.max_int, self.boundaryCellMark, self.cellsOnCell, self.nEdgesOnCell)
            self.cellInner = cellInner_tmp[:nCellsInner]
            self.cellOuter = cellOuter_tmp[:nCellsOuter]

            self.cellBoundary_ord = cmp.boundary_cells_ordered(\
                                nCellsBoundary, self.boundaryCellMark, self.cellsOnCell)

        else:
            self.cellBoundary = np.array([], dtype='int')

            
            
        
