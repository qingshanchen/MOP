import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, splu, cg, spilu, gmres
#from pysparse.itsolvers import krylov
#from pysparse.sparse import spmatrix
#from pysparse.precon import precon
#from pysparse.direct import superlu
import netCDF4 as nc
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import time
from swe_comp import swe_comp as cmp
import os
from copy import deepcopy as deepcopy 

#cmp = swe_comp.swe_comp
max_int = np.iinfo('int32').max

class parameters:

    def __init__(self):

        self.test_case = 5

        # Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
        self.timestepping = 'RK4'

        self.restart = False
        self.restart_file = 'restart.nc'

        self.bottom_topography = True
        
        self.dt = 1440.   #1440 for 480km
        self.nYears = 5/360.
        self.save_inter_days = 1
        
        self.delVisc = 0.

        # Size of the phyiscal domain
        self.earth_radius = 6371000.0
        self.Omega0 = 7.292e-5

        self.gravity = 9.81

        # Forcing
        self.tau0 = 1.e-4  #(m^2 s^-2, McWilliams et al 1981)
        self.bottomDrag = 0.e-8

        # IO files
        self.output_file = 'output.nc'

        self.nTimeSteps = np.ceil(1.*86400*360/self.dt*self.nYears).astype('int')
        self.save_interval = np.floor(1.*86400/self.dt*self.save_inter_days).astype('int')
        if self.save_interval < 1:
            self.save_interval = 1

        self.on_a_global_sphere = True

        self.use_direct_solver = False

        self.err_tol = 1e-5
        self.max_iter = 2000
        

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

            # Construct matrix for discrete Laplacian on interior cells
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
        # Poisson problem with Neumann BC's
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_neumann(self.cellsOnEdge, self.dvEdge, self.dcEdge, \
                    self.areaCell)
        #D2_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
        #                       cols[:nEntries])), shape=(self.nCells, self.nCells))
        D2s_coo = coo_matrix((valEntries[:nEntries]*self.areaCell[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(self.nCells, self.nCells))
        
        # Convert to csc sparse format
        self.D2s = D2s_coo.tocsc( )

        if c.use_direct_solver:
            self.lu_D2s = splu(self.D2s)

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
        # Poisson problem with Neumann BC's
        nEntries, rows, cols, valEntries = \
          cmp.construct_discrete_laplace_triangle_neumann(self.boundaryEdgeMark, self.verticesOnEdge, \
                      self.dvEdge, self.dcEdge, self.areaTriangle)
        E2s_coo = coo_matrix((valEntries[:nEntries]*self.areaTriangle[rows[:nEntries]], (rows[:nEntries], \
                               cols[:nEntries])), shape=(self.nVertices, self.nVertices))
        # Convert to csc sparse format
        self.E2s = E2s_coo.tocsc( )

        if c.use_direct_solver:
            self.lu_E2s = splu(self.E2s)

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

class ILU_Precon:
    """
    A preconditioner based on an
    incomplete LU factorization.

    Input: A matrix in CSR format.
    Keyword argument: Drop tolerance.
    """
    def __init__(self, A, drop=1.0e-2):
        #self.LU = superlu.factorize(A, drop_tol=drop)
        self.LU = spilu(A, drop_tol=drop)
        self.shape = self.LU.shape

    def precon(self, x, y):
        #self.LU.solve(x,y)
        y = self.LU.solve(x)


class state_data:
    def __init__(self, g, c):

        # Prognostic variables
        self.thickness = np.zeros(g.nCells)
        self.vorticity = self.thickness.copy()
        self.divergence = self.thickness.copy()

        # Diagnostic variables
        self.vorticity_vertex = np.zeros(g.nVertices)
        self.divergence_vertex = np.zeros(g.nVertices)
        self.psi_cell = np.zeros(g.nCells)
        self.psi_vertex = np.zeros(g.nVertices)
        self.phi_cell = np.zeros(g.nCells)
        self.phi_vertex = np.zeros(g.nVertices)
        self.nVelocity = np.zeros(g.nEdges)
        self.tVelocity = np.zeros(g.nEdges)
        self.pv_cell = np.zeros(g.nCells)
        self.pv_edge = np.zeros(g.nEdges)
        self.thickness_edge = np.zeros(g.nEdges)
        self.eta_cell = np.zeros(g.nCells)
        self.eta_edge = np.zeros(g.nEdges)
        self.kinetic_energy = np.zeros(g.nCells)

        # Forcing
        self.curlWind_cell = np.zeros(g.nCells)

        # Time keeper
        self.time = 0.0


    def start_from_function(self, g, c):

        latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
        lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

        latmid = 0.5*(latmin+latmax)
        latwidth = latmax - latmin

        lonmid = 0.5*(lonmin+lonmax)
        lonwidth = lonmax - lonmin

        pi = np.pi; sin = np.sin; exp = np.exp
        r = c.earth_radius

        if c.test_case == 1:
            a = c.earth_radius
            R = a/3
            u0 = 2*np.pi*a / (12*86400)
            h0 = 1000.
            lon_c = .5*np.pi
            lat_c = 0.
            r = a*np.arccos(np.sin(lat_c)*np.sin(g.latCell[:]) + \
                np.cos(lat_c)*np.cos(g.latCell[:])*np.cos(g.lonCell[:]-lon_c))
            self.thickness[:] = np.where(r<=R, 0.25*h0*(1+np.cos(np.pi*r/R)), 0.) + h0

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            self.compute_diagnostics(g, c)
            

        elif c.test_case == 2:
            a = c.earth_radius
            gh0 = 2.94e4
            u0 = 2*np.pi*a / (12*86400)
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            self.thickness[:] = gh / c.gravity

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            self.compute_diagnostics(g, c)

            if False:
                # To check that vorticity and
                psi_true = -a * u0 * np.sin(g.latCell)
                psi_vertex_true = -a * u0 * np.sin(g.latVertex)
                psi_vertex_true -= psi_true[0]
                psi_true -= psi_true[0]
                u_true = u0 * np.cos(g.latEdge)

                print("Max in nVelocity: %e" % np.max(self.nVelocity))
                print("Max in u_true: %e" % np.max(u_true))
                edgeInd = np.argmax(self.nVelocity)
                cell0 = g.cellsOnEdge[edgeInd, 0] - 1
                cell1 = g.cellsOnEdge[edgeInd, 1] - 1
                vertex0 = g.verticesOnEdge[edgeInd,0] - 1
                vertex1 = g.verticesOnEdge[edgeInd,1] - 1
                nVector = np.array([g.xCell[cell1] - g.xCell[cell0], g.yCell[cell1] - g.yCell[cell0], g.zCell[cell1] - g.zCell[cell0]])
                nVector /= np.sqrt(np.sum(nVector**2))
                hVector = np.array([-g.yEdge[edgeInd], g.xEdge[edgeInd], 0])
                hVector /= np.sqrt(np.sum(hVector**2))
                print("latEdge[%d] = %e" % (edgeInd, g.latEdge[edgeInd])) 
                print("lonEdge[%d] = %e" % (edgeInd, g.lonEdge[edgeInd])) 
                print("Actual horizontal velocity at edge %d: %e" % (edgeInd, u_true[edgeInd]))
                print("Actual normal velocity component: %e" % (u_true[edgeInd]*np.dot(nVector, hVector)))
                print("Approximate normal velocity component: %e" % (self.nVelocity[edgeInd],))
                print("Actual psi at vertex %d: %e" % (vertex0, -a*u0*np.sin(g.latVertex[vertex0]) + a*u0*np.sin(g.latCell[0])))
                print("Approximate psi at vertex %d: %e" % (vertex0, self.psi_vertex[vertex0]))
                print("Actual psi at vertex %d: %e" % (vertex1, -a*u0*np.sin(g.latVertex[vertex1]) + a*u0*np.sin(g.latCell[0])))
                print("Approximate psi at vertex %d: %e" % (vertex1, self.psi_vertex[vertex1]))
                print("dvEdge[%d] = %e" % (edgeInd, g.dvEdge[edgeInd]))
                print("")


                print("Max in tVelocity: %e" % np.max(self.tVelocity))
                print("Max in u_true: %e" % np.max(u_true))
                print("")

                print("Max in psi: %e" % np.max(self.psi_cell))
                print("Max in psi_vertex: %e" % np.max(self.psi_vertex))
                print("L-infinity error in psi: %e" % (np.max(np.abs(self.psi_cell - psi_true)) / np.max(np.abs(psi_true)),) )
                print("L-infinity error in psi_vertex: %e" % (np.max(np.abs(self.psi_vertex - psi_vertex_true)) / np.max(np.abs(psi_vertex_true)),) )
                print("")

                print("Max in phi: %e" % np.max(self.phi_cell))
                print("Max in phi_vertex: %e" % np.max(self.phi_vertex))
                print("")

                raise ValueError("Abort after testing in start_from_function")

        elif c.test_case == 5:
            a = c.earth_radius
            u0 = 20.

            h0 = 5960.
            gh = c.gravity*h0 - np.sin(g.latCell[:])**2 * (a*c.Omega0*u0 + 0.5*u0*u0) 
#            gh0 = 5960 * c.gravity
#            gh = np.sin(g.latCell[:])**2
#            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            h = gh / c.gravity

            # Define the mountain topography
            h_s0 = 2000.
            R = np.pi / 9
            lat_c = np.pi / 6.
            lon_c = .5*np.pi
            r = np.sqrt((g.latCell[:]-lat_c)**2 + (g.lonCell[:]-lon_c)**2)
            r = np.where(r < R, r, R)
            g.bottomTopographyCell[:] = h_s0 * ( 1 - r/R)
            self.thickness[:] = h[:] - g.bottomTopographyCell[:]

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            self.compute_diagnostics(g, c)

        else:
            raise ValueError("Invaid choice for the test case.")

                                                
        # Set time to zero
        self.time = 0.0
        
    def restart_from_file(self, g, c):
        rdata = nc.Dataset(c.restart_file,'r')

        start_ind = len(rdata.dimensions['Time']) - 1

        self.pv_cell[:] = rdata.variables['pv_cell'][start_ind,:,0]
        self.vorticity_cell[:] = rdata.variables['vorticity_cell'][start_ind,:,0]
        self.psi_cell[:] = rdata.variables['psi_cell'][start_ind,:,0]
        self.psi_vertex[:] = rdata.variables['psi_vertex'][start_ind,:,0]
        self.u[:] = rdata.variables['u'][start_ind,:,0]
        self.time = rdata.variables['xtime'][start_ind]
#        self.pv_edge[:] = cmp.compute_pv_edge( \
#                 g.cellsOnEdge, g.boundaryCellMark, self.pv_cell)
        self.pv_edge[:] = cmp.compute_pv_edge_apvm( \
             g.cellsOnEdge, g.boundaryCellMark, g.dcEdge, c.dt, self.pv_cell, self.u, c.apvm_factor)

        nVertices = np.size(self.psi_vertex)
        self.curlWind_cell[:] = rdata.variables['curlWind_cell'][:]

        # Read simulation parameters
        c.test_case = int(rdata.test_case)
        c.free_surface = bool(rdata.free_surface)
        c.bottom_topography = bool(rdata.bottom_topography)
        c.dt = float(rdata.dt)
        c.delVisc = float(rdata.delVisc)
        c.bottomDrag = float(rdata.bottomDrag)
        c.on_sphere = bool(rdata.on_sphere)
        c.earth_radius = float(rdata.radius)
        c.f0 = float(rdata.f0)
        
        rdata.close( )

    def initialization(self, g, c):

        if c.restart:
            self.restart_from_file(g,c)
        else:
            self.start_from_function(g, c)
            self.compute_diagnostics(g, c)
            
        # Open the output file and create new state variables
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        out.createVariable('xtime', 'f8', ('Time',))
        out.createVariable('thickness', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('vorticity_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('divergence', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('nVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('tVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('kinetic_energy', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('curlWind_cell', 'f8', ('nCells',))

        # Record parameters used for this simulation
        out.test_case = "%d" % (c.test_case)
        out.timestepping = "%s" % (c.timestepping)
        out.use_direct_solver = "%s" % (c.use_direct_solver)
        out.restart = "%s" % (c.restart)
        out.bottom_topography = "%s" % (c.bottom_topography) 
        out.dt = "%f" % (c.dt)
        out.delVisc = "%e" % (c.delVisc)
        out.bottomDrag = "%e" % (c.bottomDrag)
        out.on_a_global_sphere = "%s" % (c.on_a_global_sphere)
        out.radius = "%e" % (c.earth_radius)
        
        out.close( )


    def compute_diagnostics(self, g, c):
        # Compute diagnostic variables from pv_cell

        if c.test_case == 1:
            #For shallow water test case #1, reset the vorticity and divergence to the initial states
            a = c.earth_radius
            u0 = 2*np.pi*a / (12*86400)
            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.

        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness

        # First approach: compute psi_cell and phi_cell from vorticity and divergence, and then map them onto the dual mesh
        #self.compute_psi_cell(g, c)
        #self.compute_phi_cell(g, c)
        #self.psi_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.psi_cell)
        #self.phi_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.phi_cell)

        # Another approach: map vorticity and divergence to the dual mesh, and then compute the streamfunction and velocity potential
        # from them
        self.vorticity_vertex[:] = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.vorticity)
        self.divergence_vertex[:] = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.divergence)
        self.compute_psi_cell(g,c)
        self.compute_phi_cell(g,c)
        self.compute_psi_vertex(g,c)
        self.compute_phi_vertex(g,c)
        
        # compute the normal and tangential velocity components
        self.nVelocity = cmp.compute_normal_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_cell, self.psi_vertex)
        self.tVelocity = cmp.compute_tangential_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_vertex, self.psi_cell)

        # Map from cell to edge
        self.pv_edge[:] = cmp.cell2edge(g.cellsOnEdge, self.pv_cell)
        self.thickness_edge[:] = cmp.cell2edge(g.cellsOnEdge, self.thickness)

        # Compute absolute vorticity on edge
        self.eta_edge[:] = self.pv_edge[:] * self.thickness_edge[:]

        # Compute kinetic energy
        self.compute_kinetic_energy(g, c)


    def compute_psi_cell(self, g, c):
        # To compute the psi_cell using the elliptic equation on the
        # interior cells

        if not c.on_a_global_sphere or np.max(g.boundaryCellMark[:]) > 0:
            # A bounded domain
            self.psi_cell[:] = 0.
            x = g.lu_D1.solve(self.vorticity[g.cellInterior[:]-1])
            self.psi_cell[g.cellInterior[:]-1] = x[:]
            
        else:
            # A global domain with no boundary
            b = np.zeros(g.nCells)
            b[1:] = self.vorticity[1:]

            if c.use_direct_solver:
                #self.psi_cell[:] = g.lu_D2.solve(b)
                b *= g.areaCell[:]
                self.psi_cell[:] = g.lu_D2s.solve(b)
            elif not c.use_direct_solver:
                b *= g.areaCell[:]
                self.psi_cell = iterative_solver(g.D2s, b, self.psi_cell, c)
            else:
                raise ValueError("Indicator for director is not valid. Abort.")

        return 0


    def compute_psi_vertex(self, g, c):
        # To compute the psi_cell using the elliptic equation on the
        # interior cells

        if not c.on_a_global_sphere or np.max(g.boundaryCellMark[:]) > 0:
            # A bounded domain
            self.psi_vertex[:] = g.lu_E1.solve(self.vorticity_vertex[:])
            
        else:
            # A global domain with no boundary
            b = np.zeros(g.nVertices)
            b[1:] = self.vorticity_vertex[1:]

            if c.use_direct_solver:
                #self.psi_vertex[:] = g.lu_E2.solve(b)
                b *= g.areaTriangle[:]
                self.psi_vertex[:] = g.lu_E2s.solve(b)
            elif not c.use_direct_solver:
                b *= g.areaTriangle[:]
                self.psi_vertex = iterative_solver(g.E2s, b, self.psi_vertex, c)
            else:
                raise ValueError("Indicator for director is not valid. Abort.")
            
        return 0
    
    
    def compute_phi_cell(self, g, c):
        # To compute the phi_cell from divergence

        b = np.zeros(g.nCells)
        b[1:] = self.divergence[1:]

        if c.use_direct_solver:
            #self.phi_cell[:] = g.lu_D2.solve( b )
            b *= g.areaCell[:]
            self.phi_cell[:] = g.lu_D2s.solve( b )
        elif not c.use_direct_solver:
            b *= g.areaCell[:]
            self.phi_cell = iterative_solver(g.D2s, b, self.phi_cell, c)
        else:
            raise ValueError("Indicator for director is not valid. Abort.")

        return 0

    def compute_phi_vertex(self, g, c):
        # To compute the phi_cell from divergence

        b = np.zeros(g.nVertices)
        b[1:] = self.divergence_vertex[1:]

        if c.use_direct_solver:
            #self.phi_vertex[:] = g.lu_E2.solve( b )
            b *= g.areaTriangle[:]
            self.phi_vertex[:] = g.lu_E2s.solve( b )
        elif not c.use_direct_solver:
            b *= g.areaTriangle[:]
            self.phi_vertex = iterative_solver(g.E2s, b, self.phi_vertex, c)
        else:
            raise ValueError("Indicator for director is not valid. Abort.")

        return 0
    

    def compute_kinetic_energy(self, g, c):

        kenergy_edge = 0.5 * 0.5 * (self.nVelocity * self.nVelocity + self.tVelocity * self.tVelocity ) * g.dvEdge * g.dcEdge
        self.kinetic_energy[:] = 0.
        for iEdge in xrange(g.nEdges):
            cell0 = g.cellsOnEdge[iEdge, 0]-1
            cell1 = g.cellsOnEdge[iEdge, 1]-1
            self.kinetic_energy[cell0] += 0.5 * kenergy_edge[iEdge]
            self.kinetic_energy[cell1] += 0.5 * kenergy_edge[iEdge]

        self.kinetic_energy /= g.areaCell

        
    def save(self, c, g, k):
        # Open the output file to save current data data
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        
        out.variables['xtime'][k] = self.time
        out.variables['thickness'][k,:,0] = self.thickness[:]
        out.variables['vorticity_cell'][k,:,0] = self.vorticity[:]
        out.variables['divergence'][k,:,0] = self.divergence[:]
        out.variables['nVelocity'][k,:,0] = self.nVelocity[:]
        out.variables['tVelocity'][k,:,0] = self.tVelocity[:]

        if k==0:
            out.variables['curlWind_cell'][:] = self.curlWind_cell[:]

        self.compute_kinetic_energy(g, c)
        out.variables['kinetic_energy'][k,:,0]= self.kinetic_energy[:]
        
        out.close( )
        
def iterative_solver(A, b, x0, c):
    x, err = cg(A, b, x0=x0, tol=c.err_tol, maxiter=c.max_iter)

    if err > 0:
        raise ValueError("Convergence not achieved after %d iterations in compute_psi_cell." % err)
    elif err < 0:
        raise ValueError("Something is wrong in iterative_solver. Program abort.")
    else:
        return x

def pysparse_iterative_solver(A, b, x, K, c):

    info, iter, relres = krylov.gmres(A, b, x, c.err_tol, c.max_iter, K)

    if info < 0:
        raise ValueError("Convergence not achieved after %d iterations. Error code %d." % (iter, info))
    else:
        print("x[1001] = %e" % x[1001])
        print("iter = %d" % iter)
        print("relres = %e" % relres)
        return info
    
def timestepping_rk4_z_hex(s, g, c):

    coef = np.array([0., .5, .5, 1.])
    accum = np.array([1./6, 1./3, 1./3, 1./6])

    dt = c.dt

    s_pre = deepcopy(s)
    s_intm = deepcopy(s)

    # Update the time stamp first
    s.time += dt

    for i in xrange(4):

        # Compute the tendencies
        thicknessTransport = s_intm.thickness_edge[:] * s_intm.nVelocity[:]
        tend_thickness = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, thicknessTransport)
        
        absVorTransport = s_intm.eta_edge[:] * s_intm.nVelocity[:]
        tend_vorticity = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorTransport)
        
        absVorCirc = s_intm.eta_edge[:] * s_intm.tVelocity[:]
        geoPotent = c.gravity * (s_intm.thickness[:] + g.bottomTopographyCell[:])  + s_intm.kinetic_energy[:]
        tend_divergence = cmp.discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorCirc) - \
                          cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, geoPotent)


        # Accumulating the change in s
        s.thickness[:] += tend_thickness[:]*accum[i]*dt
        s.vorticity[:] += tend_vorticity[:]*accum[i]*dt
        s.divergence[:] += tend_divergence[:]*accum[i]*dt

        if i < 3:
            # Advance pv_cell_intm 
            s_intm.thickness[:] = s_pre.thickness[:] + coef[i+1]*dt*tend_thickness[:]
            s_intm.vorticity[:] = s_pre.vorticity[:] + coef[i+1]*dt*tend_vorticity[:]
            s_intm.divergence[:] = s_pre.divergence[:] + coef[i+1]*dt*tend_divergence[:]

            s_intm.compute_diagnostics(g, c)

    s.compute_diagnostics(g, c)


def run_tests(g, c, s):

    if False:   # Test the linear solver the Lapace equation on the interior cells with homogeneous Dirichlet BC's
        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[g.cellBoundary[:]-1] = 0.0

        vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)

        #compte psi_cell using g.A and linear solver
        x = g.lu_D1.solve(vorticity_cell[g.cellInterior[:]-1])
        psi_cell = np.zeros(g.nCells)
        psi_cell[g.cellInterior[:]-1] = x[:]

        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print "Errors for linear solver"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2        
        
    if False:
        # Test the linear solver the Lapace equation on the whole domain
        # The solution is set to zero at cell 0.
        # Also test the linear solver for the Poisson equaiton  on a bounded domain with
        # homogeneous Neumann BC's

        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[0] = 0.
        
        vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)

        # Artificially set vorticity_cell[0] to 0
        vorticity_cell[0] = 0.

        #compte psi_cell using g.A and linear solver
        psi_cell = g.lu_D2.solve(vorticity_cell[:])

        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print "Errors for linear solver"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2        

    if False:   # Test the linear solver for the Poisson equation on the triangles with homogeneous Dirichlet BC's
        psi_vertex_true = np.random.rand(g.nVertices)

        vorticity_vertex = cmp.discrete_laplace_vertex(g.verticesOnEdge,  \
                         g.dcEdge, g.dvEdge, g.areaTriangle, psi_vertex_true, 0)

        #compte psi_vertex using linear solver
        psi_vertex = g.lu_E1.solve(vorticity_vertex)

        # Compute the errors
        l8 = np.max(np.abs(psi_vertex_true[:] - psi_vertex[:])) / np.max(np.abs(psi_vertex_true[:]))
        l2 = np.sum(np.abs(psi_vertex_true[:] - psi_vertex[:])**2 * g.areaTriangle[:])
        l2 /=  np.sum(np.abs(psi_vertex_true[:])**2 * g.areaTriangle[:])
        l2 = np.sqrt(l2)
        print "Errors for the solver for the Poisson with Neumann BC's"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2        

    if False:
        # Test the linear solver for the Poisson equation on the triangles with homogeneous Neumann BC's
        # It also test the solver for the Poisson equation on the entire globe, with a zero value on triangle #0.
        psi_vertex_true = np.random.rand(g.nVertices)
        psi_vertex_true[0] = 0.

        vorticity_vertex = cmp.discrete_laplace_vertex(g.verticesOnEdge,  \
                            g.dcEdge, g.dvEdge, g.areaTriangle, psi_vertex_true, 1)

        b = vorticity_vertex[:]
        b[0] = 0.

        #compte psi_vertex using linear solver
        psi_vertex = g.lu_E2.solve(vorticity_vertex)

        # Compute the errors
        l8 = np.max(np.abs(psi_vertex_true[:] - psi_vertex[:])) / np.max(np.abs(psi_vertex_true[:]))
        l2 = np.sum(np.abs(psi_vertex_true[:] - psi_vertex[:])**2 * g.areaTriangle[:])
        l2 /=  np.sum(np.abs(psi_vertex_true[:])**2 * g.areaTriangle[:])
        l2 = np.sqrt(l2)
        print "Errors for the solver for the Poisson with Neumann BC's"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2        

    if True:
        b = np.zeros(g.nCells)
        b[1:] = np.random.rand(g.nCells-1) * 1.e-5

        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        x1[:] = g.lu_D2.solve(b)
        t1 = time.clock( )
        print("CPU time for the direct method: %f" % (t1-t0,))
        
        b *= g.areaCell[:]
        Kilu = ILU_Precon(g.D2s, drop=1.e-3)

        t0 = time.clock( )
        x1a = np.zeros(g.nCells)
        x1a, info = cg(g.D2s, b, x1a, tol=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("CPU time for scipy cg solver: %f" % (t1-t0,))

        t0 = time.clock( )
        x1b = np.zeros(g.nCells)
        x1b, info = gmres(g.D2s, b, None, tol=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("CPU time for scipy gmres solver: %f" % (t1-t0,))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        info, iter, relres = krylov.pcg(g.D2s_ps, b, x2, c.err_tol, c.max_iter)
        t1 = time.clock( )
        print("info = %d" % info)
        print("iteration # = %d" % iter)
        print("relres = %e" % relres)
        print("CPU time for pysparse pcg: %f" % (t1-t0,))

        t0 = time.clock( )
        x3 = np.zeros(g.nCells)
        info, iter, relres = krylov.gmres(g.D2s_ps, b, x3, c.err_tol, c.max_iter)
        t1 = time.clock( )
        print("info = %d" % info)
        print("iteration # = %d" % iter)
        print("relres = %e" % relres)
        print("CPU time for pysparse gmres: %f" % (t1-t0,))
        
        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        info, iter, relres = krylov.bicgstab(g.D2s_ps, b, x4, c.err_tol, c.max_iter)
        t1 = time.clock( )
        print("info = %d" % info)
        print("iteration # = %d" % iter)
        print("relres = %e" % relres)
        print("CPU time for pysparse bicgstab solver: %f" % (t1-t0,))

        t0 = time.clock( )
        x5 = np.zeros(g.nCells)
        info, iter, relres = krylov.qmrs(g.D2s_ps, b, x5, c.err_tol, c.max_iter)
        t1 = time.clock( )
        print("info = %d" % info)
        print("iteration # = %d" % iter)
        print("relres = %e" % relres)
        print("CPU time for pysparse qmrs solver: %f" % (t1-t0,))

        t0 = time.clock( )
        x6 = np.zeros(g.nCells)
        info, iter, relres = krylov.minres(g.D2s_ps, b, x6, c.err_tol, c.max_iter)
        t1 = time.clock( )
        print("info = %d" % info)
        print("iteration # = %d" % iter)
        print("relres = %e" % relres)
        print("CPU time for pysparse minres solver: %f" % (t1-t0,))

        raise ValueError
        
        
def main( ):


    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

    c = parameters()
    g = grid_data('grid.nc', c)
    s = state_data(g, c)

    #run_tests(g, c, s)
    #raise ValueError("Just for testing.")

    s.initialization(g, c)

    s_init = deepcopy(s)
    h0 = np.mean(s_init.thickness[:])
        
    # Compute energy and enstrophy
    kenergy = np.zeros(c.nTimeSteps+1)
    penergy = np.zeros(c.nTimeSteps+1)
    total_energy = np.zeros(c.nTimeSteps+1)
    mass = np.zeros(c.nTimeSteps+1)
    penstrophy = np.zeros(c.nTimeSteps+1)

    kenergy[0] = np.sum(s.kinetic_energy[:]*g.areaCell[:])
    penergy[0] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
    total_energy[0] = kenergy[0] + penergy[0]
    mass[0] = np.sum(s.thickness[:] * g.areaCell[:])
    penstrophy[0] = 0.5 * np.sum(g.areaCell[:] * s.pv_cell[:]**2)


    print("Running test case \#%d" % c.test_case)
    print("K-nergy, p-energy, t-energy, p-enstrophy, mass: %e, %e, %e, %e, %e" % (kenergy[0], penergy[0], total_energy[0], penstrophy[0], mass[0]))

    error1 = np.zeros((c.nTimeSteps+1, 3)); error1[0,:] = 0.
    error2 = np.zeros((c.nTimeSteps+1, 3)); error2[0,:] = 0.
    errorInf = np.zeros((c.nTimeSteps+1, 3)); errorInf[0,:] = 0.

    s.save(c, g, 0)

    # Entering the loop
    t0 = time.clock( )
    t0a = time.time( )
    for iStep in xrange(c.nTimeSteps):

        print "Doing step %d " % iStep

        if c.timestepping == 'RK4':
            timestepping_rk4_z_hex(s, g, c)
        else:
            raise ValueError("Invalid value for timestepping")

        # Compute energy and enstrophy
        kenergy[iStep+1] = np.sum(s.kinetic_energy[:]*g.areaCell[:])
        penergy[iStep+1] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
        total_energy[iStep+1] = kenergy[iStep+1] + penergy[iStep+1]
        mass[iStep+1] = np.sum(s.thickness[:] * g.areaCell[:])
        penstrophy[iStep+1] = 0.5 * np.sum(g.areaCell[:] * s.pv_cell[:]**2)
        print("K-nergy, p-energy, t-energy, p-enstrophy, mass: %e, %e, %e, %e, %e" % \
              (kenergy[iStep+1], penergy[iStep+1], total_energy[iStep+1], penstrophy[iStep+1], mass[iStep+1]))

        if kenergy[iStep+1] != kenergy[iStep+1]:
            print "Exceptions detected in energy. Stop now"
            raise ValueError 
        
        if np.mod(iStep+1, c.save_interval) == 0:
            k = (iStep+1) / c.save_interval
            s.save(c,g,k)

        if c.test_case == 2:
            # For test case #2, compute the errors
            error1[iStep+1, 0] = np.sum(np.abs(s.thickness[:] - s_init.thickness[:])*g.areaCell[:]) / np.sum(np.abs(s_init.thickness[:])*g.areaCell[:])
            error1[iStep+1, 1] = np.sum(np.abs(s.vorticity[:] - s_init.vorticity[:])*g.areaCell[:]) / np.sum(np.abs(s_init.vorticity[:])*g.areaCell[:])
            error1[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:])) 

            error2[iStep+1, 0] = np.sqrt(np.sum((s.thickness[:] - s_init.thickness[:])**2*g.areaCell[:]))
            error2[iStep+1,0] /= np.sqrt(np.sum((s_init.thickness[:])**2*g.areaCell[:]))
            error2[iStep+1, 1] = np.sqrt(np.sum((s.vorticity[:] - s_init.vorticity[:])**2*g.areaCell[:]))
            error2[iStep+1,1] /= np.sqrt(np.sum((s_init.vorticity[:])**2*g.areaCell[:]))
            error2[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:])) 

            errorInf[iStep+1, 0] = np.max(np.abs(s.thickness[:] - s_init.thickness[:])) / np.max(np.abs(s_init.thickness[:]))
            errorInf[iStep+1, 1] = np.max(np.abs(s.vorticity[:] - s_init.vorticity[:])) / np.max(np.abs(s_init.vorticity[:]))
            errorInf[iStep+1, 2] = np.max(np.abs(s.divergence[:] - s_init.divergence[:])) 

    days = c.dt * np.arange(c.nTimeSteps+1) / 86400.
    t1 = time.clock( )
    t1a = time.time( )
    plt.close('all')

    plt.figure(0)
    plt.plot(days, kenergy, '--', label="Kinetic energy", hold=True)
#    plt.plot(days, penergy, '-.', label="Potential energy")
#    plt.plot(days, total_energy, '-', label="Total energy")
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    plt.ylim(2.5e17, 2.6e17)
    plt.legend(loc=1)
    plt.savefig('energy.pdf', format='PDF')

    plt.figure(6)
#    plt.plot(days, kenergy, '--', label="Kinetic energy", hold=True)
    plt.plot(days, penergy, '-.', label="Potential energy", hold=True)
    plt.plot(days, total_energy, '-', label="Total energy")
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    plt.ylim(8.0e20,8.15e20)
    plt.legend(loc=1)
    plt.savefig('total-energy.pdf', format='PDF')
    
    plt.figure(1)
    plt.plot(days, penstrophy)
    plt.xlabel('Time (days)')
    plt.ylabel('Enstrophy')
    plt.ylim(0.74, 0.78)
    plt.savefig('enstrophy.pdf', format='PDF')

    plt.figure(5)
    plt.plot(days, mass)
    plt.xlabel('Time (days)')
    plt.ylabel('Mass')
    plt.ylim(1.175e18, 1.225e18)
    plt.savefig('mass.pdf', format='PDF')
    
    if c.test_case == 2:
        plt.figure(2); 
        plt.plot(days, error1[:,0], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,0], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,0], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-h.pdf', format='PDF')

        plt.figure(3); 
        plt.plot(days, error1[:,1], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,1], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,1], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-vorticity.pdf', format='PDF')

        plt.figure(4); 
        plt.plot(days, error1[:,2], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,2], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,2], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Absolute error')
        plt.savefig('error-divergence.pdf', format='PDF')

        print("Final l2 errors for thickness, vorticity, and divergence:")
        print("                    %e,        %e,     %e" % (error2[-1,0], error2[-1,1], error2[-1,2]))
        

    print 'CPU time used: %f seconds' % (t1-t0)
    print 'Walltime used: %f seconds' % (t1a-t0a)

        
if __name__ == '__main__':
    main( )


            
        
    
