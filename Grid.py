import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu
import netCDF4 as nc
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


#        if c.on_a_global_sphere:
#            self.boundaryEdgeMark = np.zeros(self.nEdges).astype('int32')
#            self.boundaryEdgeMark[:] = 0
#            self.boundaryCellMark = np.zeros(self.nCells).astype('int32')
#            self.boundaryCellMark[:] = 0
#        else:
#            self.boundaryEdgeMark = grid.variables['boundaryEdgeMark'][:]
#            self.boundaryCellMark = grid.variables['boundaryCellMark'][:]

        self.boundaryEdgeMark = grid.variables['boundaryEdgeMark'][:]
        self.boundaryCellMark = grid.variables['boundaryCellMark'][:]
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
        self.bottomTopographyCell = np.zeros(self.nCells)
        self.bottomTopographyVertex = np.zeros(self.nVertices)
        self.areaEdge = self.dvEdge * self.dcEdge / 2.

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
        out.variables['areaCell'][:] = self.areaCell[:]
        out.variables['areaTriangle'][:] = self.areaTriangle[:]
        out.variables['kiteAreasOnVertex'][:] = self.kiteAreasOnVertex[:]

        out.close( )

            
            
        
