import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, splu
import netCDF4 as nc
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import time
from swe_comp import swe_comp as cmp
import os

#cmp = swe_comp.swe_comp
max_int = np.iinfo('int32').max

class parameters:

    def __init__(self):

        self.test_case = 1

        # Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
        self.timestepping = 'RK4'

        self.restart = False
        self.restart_file = 'restart.nc'

        self.free_surface = True
        self.bottom_topography = True
        
        self.dt = 10800
        self.nYears = 10
        self.save_inter_days = 5
        
        self.delVisc = 100.

        # Size of the phyiscal domain
        self.earth_radius = 6371000.0
        self.L = 1000000.0
        self.D = np.sqrt(3.)/2*self.L
        self.H = 4000.

        self.gravity = 9.81
        self.f0 = 7.2722e-5
        self.beta = 1.4e-11

        # Forcing
        self.tau0 = 1.e-4  #(m^2 s^-2, McWilliams et al 1981)
        self.bottomDrag = 0.e-8

        # IO files
        self.output_file = 'output.nc'

        self.nTimeSteps = np.ceil(1.*86400*360/self.dt*self.nYears).astype('int')
        self.save_interval = np.ceil(1.*86400/self.dt*self.save_inter_days).astype('int')
        
        

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
        self.xCell = grid.variables['xCell'][:]
        self.yCell = grid.variables['yCell'][:]
        self.zCell = grid.variables['zCell'][:]
        self.xEdge = grid.variables['xEdge'][:]
        self.yEdge = grid.variables['yEdge'][:]
        self.latCell = grid.variables['latCell'][:]
        self.lonCell = grid.variables['lonCell'][:]
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
        self.fCell = grid.variables['fCell'][:]; #self.fCell[:] = c.f0

        self.betaY = self.fCell - c.f0
        
        if c.free_surface:
            c.fsc = c.f0 / c.H
        else:
            c.fsc = 0.0

        if c.bottom_topography:
            c.btc = c.f0 / c.H
        else:
            c.btc = 0.0
        
        self.boundaryEdgeMark = grid.variables['boundaryEdgeMark'][:] 
        self.boundaryCellMark = grid.variables['boundaryCellMark'][:] 

        # Create new grid_data variables
        self.bottomTopographyCell = np.zeros(self.nCells)
        self.bottomTopographyVertex = np.zeros(self.nVertices)
        self.bottomTopographyEdge = cmp.compute_scalar_edge(self.cellsOnEdge, self.boundaryCellMark, self.bottomTopographyCell)

        
        grid.close()

        
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
        D2_coo = coo_matrix((valEntries[:nEntries], (rows[:nEntries], \
                               cols[:nEntries])), shape=(self.nCells, self.nCells))
        # Convert to csc sparse format
        self.D2 = D2_coo.tocsc( )

        self.lu_D2 = splu(self.D2)

        raise ValueError
    
        # Make a copy of grid file
        os.system('cp %s %s' % (netcdf_file, c.output_file))

        # Open the output file to save scaled grid data
        out = nc.Dataset(c.output_file, 'a', format='NETCDF4_CLASSIC')
        out.variables['dvEdge'][:] = self.dvEdge[:]
        out.variables['dcEdge'][:] = self.dcEdge[:]
        out.variables['areaCell'][:] = self.areaCell[:]
        out.variables['areaTriangle'][:] = self.areaTriangle[:]
        out.variables['kiteAreasOnVertex'][:] = self.kiteAreasOnVertex[:]

        out.close( )

                                                  
class state_data:
    def __init__(self, g, c):

        # Prognostic variables
        self.pv_cell = np.zeros(g.nCells)
        self.pv_vertex = np.zeros(g.nVertices)

        # Diagnostic variables
        self.u = np.zeros(g.nEdges)
        self.pv_edge = np.zeros(g.nEdges)
        self.psi_cell = np.zeros(g.nCells)
        self.psi_vertex = np.zeros(g.nVertices)
        self.vorticity_cell = np.zeros(g.nCells)
        self.kinetic_energy = np.zeros(g.nCells)

        # Forcing
        self.curlWind_cell = np.zeros(g.nCells)

        # Temporary variables
        self.pv_vertex_pre = np.zeros(g.nVertices);
        self.pv_vertex_intm = self.pv_vertex_pre.copy( )
        self.tend_pv_cell = np.zeros(g.nCells)
        self.tend_eta_cell = np.zeros(g.nCells)
        self.tend_vorticity_cell = np.zeros(g.nCells)
        self.vorticity_edge = np.zeros(g.nEdges)
        self.eta_cell = np.zeros(g.nCells)
        self.eta_edge = np.zeros(g.nEdges)

        #
        self.time = 0.0

        ### Compute the upward or downward shifting factor by solving an elliptic
        ### problem with the rhs equal to 1.
        # Set the homogeneious boundary conditon for psi_cell
        self.psi_cell[g.cellBoundary[:]-1] = 0.0

        # Compute the right-hand side
        nCellsInterior = np.size(g.cellInterior)
        b = np.zeros(nCellsInterior)
        b[:] = 1.0

        # Enforcing the boundary condition
        b -= g.D1_bdry*self.psi_cell[g.cellBoundary[:]-1]

        # Solve the linear system
        x = g.lu.solve(b)

        # Recover psi from x
        self.psi_cell[g.cellInterior[:]-1] = x[:].copy()

        # Shift psi so that area-weighted average of psi is zero (to preserve volume)
        psi_avg = np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)

        # Compute the shifting factor
        self.shifting_factor = 1 + c.fsc * psi_avg

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
            # Single-gyre
            self.psi_cell[:] = 0.0
            self.psi_vertex[:] = 0.0
            #self.compute_pv_cell(c, g)
            self.pv_cell[:] = g.betaY[:] + c.btc * g.bottomTopographyCell[:]
            self.eta_cell[:] = 0.
            
            # Initialize wind
            # wind = -tau0 * cos(pi*(y-ymid)/L)  (see Greatbatch and Nag)  
            self.curlWind_cell[:] = -c.tau0 * np.pi/(latwidth*r) * \
                                    np.sin(np.pi*(g.latCell[:]-latmin) / latwidth)

            
        elif c.test_case == 2:
            # fluid at rest, and pv equals the planetary vorticity
            self.pv_cell = g.betaY[:]
            self.compute_psi_cell(g, c)
            self.psi_vertex[:] = cmp.cell2vertex( \
                g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, \
                g.verticesOnEdge, self.psi_cell)

            # Initialize wind
            # wind = tau0 * cos(2*pi*(y-ymid)/L)  (see Greatbatch and Nag)  
            self.curlWind_cell[:] = 0.0

            # Eliminate bottom drag
            c.bottomDrag = 0.

            # Eliminate lateral diffusion
            c.delVisc = 0.

        elif c.test_case == 3:
            # Test case with a homongenized PV field, zero wind and zero bottom drag.
            # There is no diffusion either
            
            self.pv_cell = np.zeros(g.nCells)
            self.pv_cell[:] = 0.
            self.compute_psi_cell(g, c)
            self.psi_vertex[:] = cmp.cell2vertex( \
                g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, \
                g.verticesOnEdge, self.psi_cell)

            # Initialize wind
            # wind = tau0 * cos(2*pi*(y-ymid)/L)  (see Greatbatch and Nag)  
            self.curlWind_cell[:] = 0.0

            # Eliminate bottom drag
            c.bottomDrag = 0.

            # Eliminate lateral diffusion
            c.delVisc = 0.

        elif c.test_case == 4:
            # One gyre with no forcing and drag
            d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-.85))**2/.36**2)
            self.psi_cell[:] = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
            self.psi_cell[:] -= np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
            self.psi_vertex[:] = cmp.cell2vertex( \
                g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, \
                g.verticesOnEdge, self.psi_cell)

            # Compute pv_cell
            vorticity_cell = cmp.discrete_laplace_cell( \
                 g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                 self.psi_cell)
            vorticity_cell *= c.gravity / c.f0
            self.pv_cell = vorticity_cell
            self.pv_cell += g.betaY
            self.pv_cell -=  c.fsc*self.psi_cell
            self.pv_cell += c.btc*g.bottomTopographyCell
            
            #self.compute_pv_cell(c, g)
            
            # Initialize wind
            self.curlWind_cell[:] = 0.

            # Eliminate bottom drag
            c.bottomDrag = 0.

            # Eliminate lateral diffusion
            c.delVisc = 0.
            
        elif c.test_case == 9:
            #Double gyre
            self.psi_vertex[:] = 0.0
            self.psi_cell = cmp.update_psi_cell(g.boundaryVertex[:,0], \
                                g.cellsOnVertex, g.kiteAreasOnVertex, \
                                g.areaCell, self.psi_vertex)

            # Initialize wind
            # wind = tau0 * cos(2*pi*(y-ymid)/L)  (see Greatbatch and Nag)  
            self.curlWind_cell[:] = 4*c.tau0 * np.pi/(latwidth*r) * \
                                    np.sin(4*np.pi*(g.latCell[:]-latmid) / latwidth)
            self.curlWind_cell = np.where(g.latCell > latmid + latwidth/4, 0., self.curlWind_cell[:])
            self.curlWind_cell = np.where(g.latCell < latmid - latwidth/4, 0., self.curlWind_cell[:])
                                                
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
        out.createVariable('u', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('pv_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('pv_vertex', 'f8', ('Time', 'nVertices', 'nVertLevels'))
        out.createVariable('vorticity_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('vorticity_vertex', 'f8', ('Time', 'nVertices', 'nVertLevels'))
        out.createVariable('psi_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('psi_vertex', 'f8', ('Time', 'nVertices', 'nVertLevels'))
        out.createVariable('kinetic_energy', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('curlWind_cell', 'f8', ('nCells',))

        # Record parameters used for this simulation
        out.test_case = "%d" % (c.test_case)
        out.timestepping = "%s" % (c.timestepping)
        out.restart = "%s" % (c.restart)
        out.free_surface = "%s" % (c.free_surface) 
        out.bottom_topography = "%s" % (c.bottom_topography) 
        out.dt = "%f" % (c.dt)
        out.delVisc = "%e" % (c.delVisc)
        out.bottomDrag = "%e" % (c.bottomDrag)
        out.on_sphere = "True"
        out.radius = "%e" % (c.earth_radius)
        out.f0 = "%e" % (c.f0)
        
        out.close( )


    def compute_diagnostics(self, g, c):
        # Compute diagnostic variables from pv_cell

        self.compute_psi_cell(g, c)

        #self.compute_vorticity_cell(g, c)
        # No-slip, mirroring BC
        self.vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                                                 self.psi_cell)
        self.vorticity_cell *= c.gravity / c.f0

        # If lateral diffusion is absent, enforce the free-slip BC's
        if c.delVisc < np.finfo('float32').tiny:
            self.vorticity_cell[g.cellBoundary[:]-1] = 0.

        # Update pv_cell on the boundary
        bcs = g.cellBoundary[:]-1
        self.pv_cell[bcs] = self.vorticity_cell[bcs] + g.betaY[bcs] - c.fsc*self.psi_cell[bcs] + c.btc * g.bottomTopographyCell[bcs]
        
        # Average psi_cell to psi_vertex
        self.psi_vertex = cmp.cell2vertex( \
            g.cellsOnVertex, g.kiteAreasOnVertex, \
            g.areaTriangle, g.verticesOnEdge, self.psi_cell)

        self.u[:] = cmp.compute_u(g.verticesOnEdge, g.cellsOnEdge, g.dvEdge, \
                        self.psi_vertex, self.psi_cell, c.gravity/c.f0)

        self.pv_edge[:] = cmp.compute_pv_edge( \
            g.cellsOnEdge, g.boundaryCellMark, self.pv_cell)
#        self.pv_edge[:] = cmp.compute_pv_edge_apvm( \
#             g.cellsOnEdge, g.boundaryCellMark, g.dcEdge, c.dt, self.pv_cell, self.u, c.apvm_factor)

        
    def compute_vorticity_cell(self, g, c):
        self.vorticity_cell[:] = self.pv_cell[:] - g.betaY[:]
        self.vorticity_cell[:] += c.fsc * self.psi_cell[:]
        self.vorticity_cell[:] -= c.btc * g.bottomTopographyCell[:]


    def compute_psi_cell(self, g, c):
        # To compute the psi_cell using the elliptic equation on the
        # interior cells

        # Set the homogeneious boundary conditon for psi_cell
        self.psi_cell[g.cellBoundary[:]-1] = 0.0

        ## Compute the right-hand side
        nCellsInterior = np.size(g.cellInterior)
        b = np.zeros(nCellsInterior)
        b[:] = self.pv_cell[g.cellInterior[:]-1]

        # Subtract bottom topography
        b -= c.btc * g.bottomTopographyCell[g.cellInterior[:]-1]

        # Subtract planetary vorticity
        b -= g.betaY[g.cellInterior[:]-1] 

        # Enforcing the boundary condition
        b -= g.D1_bdry*self.psi_cell[g.cellBoundary[:]-1]

        # Solve the linear system
        x = g.lu.solve(b)

        # Recover psi from x
        self.psi_cell[g.cellInterior[:]-1] = x[:].copy()

        # Compute the constant boundary value for psi satsifying the constraint that
        # integral(psi) over Omega is zero.
        psi_avg = np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
        const_bdry_valu = -psi_avg / self.shifting_factor

        # Add the shifting to the right-hand side
        b += c.fsc * const_bdry_valu

        # Solve the boundary value again
        x = g.lu.solve(b)

        # Recover psi from x
        self.psi_cell[g.cellInterior[:]-1] = x[:].copy()

        # Add the shifting to psi
        self.psi_cell += const_bdry_valu

        if self.psi_cell.max( ) != self.psi_cell.max( ):
            raise ValueError("Exceptions detected in compute_psi_cell")

        return 0

    
    def compute_pv_cell(self, c, g):
        self.pv_cell[:] = self.vorticity_cell[:] +  g.betaY[:]
        self.pv_cell[:] -= c.fsc* self.psi_cell
        self.pv_cell[:] += c.btc * g.bottomTopographyCell[:]


    def compute_kinetic_energy(self, c, g):

        kenergy_edge = 0.5 * 0.5*(self.u * self.u ) * g.dvEdge * g.dcEdge
        self.kinetic_energy[:] = 0.
        for iEdge in xrange(g.nEdges):
            cell0 = g.cellsOnEdge[iEdge, 0]-1
            cell1 = g.cellsOnEdge[iEdge, 1]-1
            self.kinetic_energy[cell0] += kenergy_edge[iEdge]
            self.kinetic_energy[cell1] += kenergy_edge[iEdge]

        self.kinetic_energy /= g.areaCell

        
    def save(self, c, g, k):
        # Open the output file to save current data data
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        
        out.variables['xtime'][k] = self.time
        out.variables['u'][k,:,0] = self.u[:]
        out.variables['pv_cell'][k,:,0] = self.pv_cell[:]
        out.variables['pv_vertex'][k,:,0] = self.pv_vertex[:]
        out.variables['psi_cell'][k,:,0] = self.psi_cell[:]
        out.variables['psi_vertex'][k,:,0] = self.psi_vertex[:]
        out.variables['vorticity_cell'][k,:,0] = self.vorticity_cell[:]

        if k==0:
            out.variables['curlWind_cell'][:] = self.curlWind_cell[:]

        self.compute_kinetic_energy(c, g)
        out.variables['kinetic_energy'][k,:,0]= self.kinetic_energy[:]
        
        out.close( )
        

def timestepping_rk4_z_hex(s, s_pre, s_intm, g, c):

    coef = np.array([0., .5, .5, 1.])
    accum = np.array([1./6, 1./3, 1./3, 1./6])

    dt = c.dt

    s_pre.pv_cell[:] = s.pv_cell[:]
    s_intm.pv_cell[:] = s.pv_cell[:]
    s_intm.pv_edge[:] = s.pv_edge[:]
    s_intm.vorticity_cell[:] = s.vorticity_cell[:]
    s_intm.u[:] = s.u[:]
    s_pre.psi_cell[:] = s.psi_cell[:]
    s_pre.psi_vertex[:] = s.psi_vertex[:]

    # Update the time stamp first
    s.time += dt

    for i in xrange(4):

        ## Advance the vorticity equation
        # Compute tend_pv_cell 
        s.tend_pv_cell[:] = \
          cmp.compute_tend_pv_cell( \
            g.boundaryEdgeMark[:], g.cellsOnEdge[:,:], s_intm.pv_edge, \
            s_intm.vorticity_cell, s_intm.u, g.dcEdge, g.dvEdge, g.areaCell, \
                      s.curlWind_cell, c.bottomDrag, c.delVisc, c.H)

        # Accumulating the change in pv_cell and pv_vertex fields
        s.pv_cell[:] += s.tend_pv_cell[:]*accum[i]*dt

        if s.pv_cell[g.cellInterior[:]-1].max( ) != s.pv_cell[g.cellInterior[:]-1].max():
            raise ValueError("Exceptions detected in pv_cell")

        if i < 3:
            # Advance pv_cell_intm 
            s_intm.pv_cell[:] = s_pre.pv_cell[:] + coef[i+1]*dt*s.tend_pv_cell[:]

            s_intm.compute_diagnostics(g, c)

    s.compute_diagnostics(g, c)


def timestepping_euler(s, s_pre, g, c):

    delVisc = c.delVisc
    del2Visc = c.del2Visc
    dt = c.dt

    s_pre.pv_cell[:] = s.pv_cell[:]
    s_pre.pv_edge[:] = s.pv_edge[:]
    s_pre.vorticity_cell[:] = s.vorticity_cell[:]
    s_pre.u[:] = s.u[:]
    s_pre.psi_cell[:] = s.psi_cell[:]
    s_pre.psi_vertex[:] = s.psi_vertex[:]

    # Update the stime stamp first
    s.time += dt

    s.tend_pv_cell[:] = cmp.compute_tend_pv_cell( \
        g.boundaryEdgeMark[:], g.cellsOnEdge[:,:], s_pre.pv_edge, \
        s_pre.vorticity_cell, s_pre.u, g.dcEdge, g.dvEdge, g.areaCell, \
                s.curlWind_cell, c.bottomDrag, c.delVisc, del2Visc, c.H)

    # Accumulating the change in pv_cell and pv_vertex fields
    s.pv_cell[:] += s.tend_pv_cell[:]*dt

    # Update the diagnostic variables
    s.compute_diagnostics(g, c)

    
def timestepping_backwardEuler(s, s_pre, g, c):

    delVisc = c.delVisc
    dt = c.dt

    ## Time keeping
    s.time += dt

    s_pre.vorticity_cell[:] = s.vorticity_cell[:]
    s_pre.pv_cell[:] = s.pv_cell[:]
    s_pre.psi_cell[:] = s.psi_cell[:]
    s_pre.pv_edge[:] = s.pv_edge[:]
    s_pre.u[:] = s.u[:]

    # Compute the right-hand side
    b = c.dt / c.H * s.curlWind_cell[g.cellInner[:] - 1]

    pv_fluxes_cell = cmp.compute_pv_fluxes_cell(g.cellsOnEdge, s_pre.pv_edge, \
                                                s_pre.u, g.dvEdge, g.areaCell)
    b -= c.dt * pv_fluxes_cell[g.cellInner[:]-1]
    b += s_pre.vorticity_cell[g.cellInner[:]-1]
    b -= c.fsc * s_pre.psi_cell[g.cellInner[:]-1]

    x = g.lu_C.solve(b)

    ## Recover psi from x
    s.psi_cell[g.cellInner[:]-1] = x[:].copy()

    ## Enforce homogeneous Dirichlet and Neumann BCs
    s.psi_cell[g.cellOuter[:]-1] = 0.

    # Compute the constant boundary value for psi
    psi_avg = np.sum(s.psi_cell * g.areaCell) / np.sum(g.areaCell)

    # Subtract the constant value from psi so that it has an average value of zero
    #s.psi_cell -= psi_avg

    # Compute vorticity_cell from psi_cell
    s.vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                                                 s.psi_cell)
    s.vorticity_cell *= c.gravity / c.f0

    # Compute pv_cell
    s.compute_pv_cell(c, g)

    # Average psi_cell to psi_vertex
    s.psi_vertex = cmp.cell2vertex( \
        g.cellsOnVertex, g.kiteAreasOnVertex, \
        g.areaTriangle, g.verticesOnEdge, s.psi_cell)

    s.u[:] = cmp.compute_u(g.verticesOnEdge, g.cellsOnEdge, g.dvEdge, \
                    s.psi_vertex, s.psi_cell, c.gravity/c.f0)

    s.pv_edge[:] = cmp.compute_pv_edge( \
         g.cellsOnEdge, g.boundaryCellMark, s.pv_cell)
    #s.pv_edge[:] = cmp.compute_pv_edge_apvm( \
    #     g.cellsOnEdge, g.boundaryCellMark, g.dcEdge, c.dt, s.pv_cell, s.u, c.apvm_factor)
    #s.pv_edge[:] = cmp.compute_pv_edge_upwind(g.cellsOnEdge, s.pv_cell, s.u)


def timestepping_steady(s, s_pre, g, c):

    dt = c.dt

    ## Time keeping
    s.time += dt

    ## Solve an elliptic BVP for psi
#    b = 1. / c.H * s.curlWind_cell[g.cellInterior[:] - 1]
#    x = g.lu_B.solve(b)
#    s.psi_cell[g.cellInterior[:]-1] = x[:].copy()
#    s.psi_cell[g.cellBoundary[:]-1] = 0.

    ## Solve an biharmonic BVP for psi
    b = 1. / c.H * s.curlWind_cell[g.cellInner[:] - 1]
    x = g.lu_B2.solve(b)
    s.psi_cell[g.cellInner[:]-1] = x[:].copy()
    s.psi_cell[g.cellOuter[:]-1] = 0.
    
    s.vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                                                 s.psi_cell)
    s.vorticity_cell *= c.gravity / c.f0

    # Compute pv_cell
    s.compute_pv_cell(c, g)

    # Average psi_cell to psi_vertex
    s.psi_vertex = cmp.cell2vertex( \
        g.cellsOnVertex, g.kiteAreasOnVertex, \
        g.areaTriangle, g.verticesOnEdge, s.psi_cell)

    s.u[:] = cmp.compute_u(g.verticesOnEdge, g.cellsOnEdge, g.dvEdge, \
                    s.psi_vertex, s.psi_cell, c.gravity/c.f0)

    # To test matrix g.Dx_augmented
    #Dx_psi_cell = g.Dx * x * c.beta * c.gravity / c.f0
#    Dx_psi_cell = g.Dx_augmented * s.psi_cell * c.beta * c.gravity / c.f0
#    s.pv_cell[g.cellInterior[:]-1] = Dx_psi_cell.copy()
#    s.pv_cell[g.cellBoundary[:]-1] = 0.
#    a = g.Dx * s.psi_cell[g.cellInterior[:]-1] + g.Dx_bdry * s.psi_cell[g.cellBoundary[:]-1]
#    b = g.Dx_augmented * s.psi_cell
#    print("max in a-b = %e" % np.max(np.abs(a-b)))
#    if np.max(np.abs(a-b)) > 1e-5:
#        raise ValueError

#    D1_psi_cell = g.D1 * x * c.bottomDrag * c.gravity / c.f0
#    s.vorticity_cell[g.cellInterior[:]-1] = D1_psi_cell.copy()
#    s.vorticity_cell[g.cellBoundary[:]-1] = 0.
    

def run_tests(g, c, s):

    if False:    # Test matrix A and discrete_laplace
        psi_cell_true = np.random.rand(g.nCells)

        psi_cell_true[g.cellBoundary[:]-1] = 0.0

        vorticity_cell_1 = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)
        vorticity_cell_1 *= c.gravity/c.f0

        #Compute pv_cell (without beta term and bottom topography)
        pv_cell_1 = vorticity_cell_1 - c.fsc*psi_cell_true

        pv_cell_2 = np.zeros(g.nCells)
        pv_cell_2[g.cellInterior[:]-1] = g.A_augmented * psi_cell_true
        
        # Compute the errors
        l8 = np.max(np.abs(pv_cell_1[g.cellInterior[:]-1] - pv_cell_2[g.cellInterior[:]-1])) / np.max(np.abs(pv_cell_1[g.cellInterior[:]-1]))
        l2 = np.sum(np.abs(pv_cell_1[g.cellInterior[:]-1] - pv_cell_2[g.cellInterior[:]-1])**2 * g.areaCell[g.cellInterior[:]-1])
        l2 /=  np.sum(np.abs(pv_cell_1[g.cellInterior[:]-1])**2 * g.areaCell[g.cellInterior[:]])
        l2 = np.sqrt(l2)
        print "Errors for linear solver"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2

        # Examine cell 174 (zero based)
        print("pv_cell_1[174] = %e" % pv_cell_1[174])
        print("pv_cell_2[174] = %e" % pv_cell_2[174])
        edges = g.edgesOnCell[174,:6]
        coefs = g.dvEdge[edges-1] / g.dcEdge[edges-1]
        cells = g.cellsOnCell[174,:6]
        print("pv_cell[174]= %e" % (c.gravity/c.f0 * (np.sum(psi_cell_true[cells-1] * coefs) - psi_cell_true[174]*np.sum(coefs))/g.areaCell[174] - c.fsc*psi_cell_true[174]))


    if False:   # Test the linear solver
        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[g.cellBoundary[:]-1] = 0.0

        vorticity_cell_1 = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)
        vorticity_cell_1 *= c.gravity/c.f0

        #Compute pv_cell (without beta term and bottom topography)
        pv_cell = vorticity_cell_1 - c.fsc*psi_cell_true

        #compte psi_cell using g.A and linear solver
        x = g.lu.solve(pv_cell[g.cellInterior[:]-1])
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
        
    if False:   # Test compute_psi_cell
        print("Test compute_psi_cell")
#        latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
#        lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

#        latmid = 0.5*(latmin+latmax)
#        latwidth = latmax - latmin

#        lonmid = 0.5*(lonmin+lonmax)
#        lonwidth = lonmax - lonmin

#        pi = np.pi; sin = np.sin; exp = np.exp
#        r = c.earth_radius

        #
#        d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-.85))**2/.36**2)
#        psi_cell_true = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
#        psi_cell_true[:] -= np.sum(psi_cell_true * g.areaCell) / np.sum(g.areaCell)

        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[g.cellBoundary[:]-1] = 0.0
        psi_cell_true[:] -= np.sum(psi_cell_true * g.areaCell) / np.sum(g.areaCell)
        
        # Compute vorticity_cell
        vorticity_cell_1 = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)
        vorticity_cell_1 *= c.gravity/c.f0

        # Compute pv_cell
        pv_cell = vorticity_cell_1 + g.betaY - c.fsc*psi_cell_true + c.btc*g.bottomTopographyCell

        # Test compute_psi_cell
        s.pv_cell[:] = pv_cell[:]
        s.compute_psi_cell(g, c)
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print "Errors for compute_psi_cell"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2
        if l2 > 1.e-7:
            raise ValueError("L2 error is too big.")

        psi_vertex = cmp.cell2vertex( g.cellsOnVertex, g.kiteAreasOnVertex,
                                      g.areaTriangle, g.verticesOnEdge, s.psi_cell)
        u = cmp.compute_u(g.verticesOnEdge, g.cellsOnEdge, \
                                      g.dvEdge, psi_vertex, s.psi_cell, c.gravity/c.f0)
        energy = 0.5*np.sum(g.dvEdge[:] * g.dcEdge[:] * u**2)
        print("After re-calculating psi_cell from pv_cell, energy = %e" % energy)

    if False:   # Test compute_psi_cell_del2
        print("Test compute_psi_del2")
        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[g.cellOuter[:]-1] = 0.0
        psi_cell_true[:] -= np.sum(psi_cell_true * g.areaCell) / np.sum(g.areaCell)

        # Compute vorticity_cell
        vorticity_cell_1 = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)
        vorticity_cell_1 *= c.gravity/c.f0

        # Compute pv_cell
        pv_cell = vorticity_cell_1 + g.betaY - c.fsc*psi_cell_true + c.btc*g.bottomTopographyCell

        # Test compute_psi_cell
        s.pv_cell[:] = pv_cell[:]
        s.compute_psi_cell_del2(g, c)
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print "Errors for compute_psi_cell"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2
        if l2 > 1.e-7:
            raise ValueError("L2 error is too big.")

    if True:
        
        print("Test matrix C and its solver.")
        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[g.cellOuter[:]-1] = 0.0

        # Compute the right-hand side
        # Compute vorticity_cell
        vorticity_cell_1 = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)
        vorticity_cell_1 *= c.gravity/c.f0
        b = (1. + c.bottomDrag*c.dt)*vorticity_cell_1

        b -= c.delVisc * c.dt * cmp.discrete_laplace_cell(g.cellsOnEdge, \
             g.dcEdge, g.dvEdge, g.areaCell, vorticity_cell_1)

        b -= c.fsc * psi_cell_true

        b = b[g.cellInner[:]-1]
        x = g.lu_C.solve(b)
        s.psi_cell[g.cellInner[:]-1] = x[:].copy()
        s.psi_cell[g.cellOuter[:]-1] = 0.
        
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print "Errors for compute_psi_cell"
        print "L infinity error = ", l8
        print "L^2 error        = ", l2

        
def main( ):

    # -----------------------------------------------------------
    # Setting parameters
    # -----------------------------------------------------------
#    nTimeSteps = 360*10; plot_interval = 900
#    save_interval = 1*30


    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

    c = parameters()
    g = grid_data('grid.nc', c)
    s = state_data(g, c)
    s_pre = state_data(g, c)
    s_intm = state_data(g, c)


    s.initialization(g,c)
        
    # Compute energy and enstrophy
    energy = np.zeros(c.nTimeSteps+1)
    enstrophy = np.zeros(c.nTimeSteps+1)
    energy[0] = 0.5*np.sum(g.dvEdge[:]*g.dcEdge[:]*(s.u[:]**2))
    enstrophy[0] = 0.5 * np.sum(g.areaCell[:] \
                * s.pv_cell[:]**2)

    print("Running test case \#%d" % c.test_case)
    print "Energy, enstrophy %e, %e" % (energy[0], enstrophy[0])

    #run_tests(g, c, s)
    #raise ValueError

    s.save(c, g, 0)

    # Entering the loop
    t0 = time.clock( )
    t0a = time.time( )
    for iStep in xrange(c.nTimeSteps):

        print "Doing step %d " % iStep

        if c.timestepping == 'E':
            timestepping_euler(s, s_pre, g, c)
        elif c.timestepping == 'BE':
            timestepping_backwardEuler(s, s_pre, g, c)
        elif c.timestepping == 'RK4':
            timestepping_rk4_z_hex(s, s_pre, s_intm, g, c)
        elif c.timestepping == 'steady':
            timestepping_steady(s, s_pre, g, c)
        else:
            raise ValueError("Invalid value for timestepping")

        #timestepping_rk4_eta(s, s_pre, s_intm, g, c)

        # Compute energy and enstrophy
        energy[iStep+1] = 0.5*np.sum(g.dvEdge[:]*g.dcEdge[:]*(s.u[:]**2))
        enstrophy[iStep+1] = 0.5 * np.sum(g.areaCell[:] \
                    * s.pv_cell[:]**2)

        print "Energy, enstrophy %e, %e" % (energy[iStep+1], enstrophy[iStep+1])

        if energy[iStep+1] != energy[iStep+1]:
            print "Exceptions detected in energy. Stop now"
            raise ValueError 
        
        if np.mod(iStep+1, c.save_interval) == 0:
            k = (iStep+1) / c.save_interval
            s.save(c,g,k)
            
    t1 = time.clock( )
    t1a = time.time( )
    plt.close('all')
    plt.figure(0)
    plt.plot(energy)
    plt.savefig('energy.png', format='PNG')
    plt.figure(1)
    plt.plot(enstrophy)
    plt.savefig('enstrophy.png', format='PNG')

    print 'CPU time used: %f seconds' % (t1-t0)
    print 'Walltime used: %f seconds' % (t1a-t0a)

if __name__ == '__main__':
    main( )


            
        
    
