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

        self.bottom_topography = True
        
        self.dt = 10800
        self.nYears = 10
        self.save_inter_days = 5
        
        self.delVisc = 100.

        # Size of the phyiscal domain
        self.earth_radius = 6371000.0

        self.gravity = 9.81

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
        self.thickness = np.zeros(g.nCells)
        self.vorticity = self.thickness.copy()
        self.divergence = self.thickness.copy()

        # Diagnostic variables
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

        # Temporary variables
        self.pv_vertex_pre = np.zeros(g.nVertices);
        self.pv_vertex_intm = self.pv_vertex_pre.copy( )
        self.tend_pv_cell = np.zeros(g.nCells)
        self.tend_eta_cell = np.zeros(g.nCells)
        self.tend_vorticity_cell = np.zeros(g.nCells)

        #
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

        # Compute psi_cell from vorticity
        self.compute_psi_cell(g, c)

        # Compute phi_cell from divergence
        self.compute_phi_cell(g, c)

        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness

        # Map from cell to vertex
        self.psi_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.psi_cell)
        self.phi_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.phi_cell)

        # compute the normal and tangential velocity components
        self.nVelocity = cmp.comute_normal_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_cell, self.psi_vertex)
        self.tVelocity = cmp.comute_tangential_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_vertex, self.psi_cell)

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

        self.psi_cell[:] = 0.
        x = g.lu_D1.solve(self.vorticity[g.cellInterior[:]-1])
        self.psi_cell[g.cellInterior[:]-1] = x[:]

        return 0

    
    def compute_phi_cell(self, g, c):
        # To compute the phi_cell from divergence

        self.divergence[0] = 0.
        self.phi_cell[:] = g.lu_D2.solve(self.divergence[:])

        return 0
    

    def compute_kinetic_energy(self, c, g):

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


def run_tests(g, c, s):

    if True:   # Test the linear solver the Lapace equation on the interior cells with homogeneous Dirichlet BC's
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
        
    if True:   # Test the linear solver the Lapace equation on the whole domain with homogeneous Neumann BC's
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
        

        
def main( ):


    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

    c = parameters()
    g = grid_data('grid.nc', c)
    s = state_data(g, c)
    s_pre = state_data(g, c)
    s_intm = state_data(g, c)

    run_tests(g, c, s)
    raise ValueError

    s.initialization(g,c)
        
    # Compute energy and enstrophy
    energy = np.zeros(c.nTimeSteps+1)
    enstrophy = np.zeros(c.nTimeSteps+1)
    energy[0] = 0.5*np.sum(g.dvEdge[:]*g.dcEdge[:]*(s.u[:]**2))
    enstrophy[0] = 0.5 * np.sum(g.areaCell[:] \
                * s.pv_cell[:]**2)


    print("Running test case \#%d" % c.test_case)
    print "Energy, enstrophy %e, %e" % (energy[0], enstrophy[0])


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


            
        
    
