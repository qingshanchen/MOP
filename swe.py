import numpy as np
from Grid import grid_data
from VectorCalculus import VectorCalculus
from LinearAlgebra import cg
import netCDF4 as nc
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import time
from swe_comp import swe_comp as cmp
import os
from copy import deepcopy as deepcopy
from pyamg import rootnode_solver

max_int = np.iinfo('int32').max

class parameters:

    def __init__(self):

        ### Parameters essential
        self.test_case = 5
        self.on_a_global_sphere = True

        ### Parameters secondary
        # Choose the time stepping technique: 'E', 'BE', 'RK4', 'Steady'
        self.timestepping = 'RK4'

        # Duration, time stepping size, saving interval
        self.dt = 360.   #1440 for 480km
#        self.dt = 360.   #360 for NA818
        self.nYears = .1/360
        self.save_inter_days = 5

        # Model configuraitons, boundary conditions
        self.delVisc = 0.  # 80 for NA818
        self.bottomDrag =  0. #5.e-8
        self.no_flux_BC = True  # Should always be on
        self.no_slip_BC = True
        
        # Solver config
        self.linear_solver = 'cg'      # lu, cg,
        self.err_tol = 1e-10
        self.max_iter = 2000
        
        self.restart = False
        self.restart_file = 'restart.nc'

        # Size of the phyiscal domain
        self.earth_radius = 6371000.0
        self.Omega0 = 7.292e-5

        self.gravity = 9.81

        # Forcing

        # IO files
        self.output_file = 'output.nc'

        self.nTimeSteps = np.ceil(1.*86400*360/self.dt*self.nYears).astype('int')
        self.save_interval = np.floor(1.*86400/self.dt*self.save_inter_days).astype('int')
        if self.save_interval < 1:
            self.save_interval = 1

        self.max_int = np.iinfo('int32').max



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
        self.thickness_vertex = np.zeros(g.nVertices)
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
        self.tend_thickness = np.zeros(g.nCells)
        self.tend_vorticity = np.zeros(g.nCells)
        self.tend_divergence = np.zeros(g.nCells)

        # Forcing
        self.curlWind_cell = np.zeros(g.nCells)
        self.divWind_cell = np.zeros(g.nCells)
        self.sfWind_cell = np.zeros(g.nCells)
        self.sfWind_vertex = np.zeros(g.nVertices)
        self.vpWind_cell = np.zeros(g.nCells)
        self.vpWind_vertex = np.zeros(g.nVertices)

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
            #self.compute_diagnostics(g, c)
            

        elif c.test_case == 2:
            a = c.earth_radius
            gh0 = 2.94e4
            u0 = 2*np.pi*a / (12*86400)
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            self.thickness[:] = gh / c.gravity

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            self.psi_cell[:] = -a * u0 * np.sin(g.latCell[:])
            self.psi_cell[:] -= self.psi_cell[0]
            self.phi_cell[:] = 0.
            self.psi_vertex[:] = -a * u0 * np.sin(g.latVertex[:])
            self.psi_vertex[:] -= self.psi_vertex[0]
            self.phi_vertex[:] = 0.
            #self.compute_diagnostics(g, c)

            if False:
                # To check that vorticity and
                psi_true = -a * u0 * np.sin(g.latCell)
                psi_vertex_true = -a * u0 * np.sin(g.latVertex)
                psi_vertex_true -= psi_true[0]
                psi_true -= psi_true[0]
                u_true = u0 * np.cos(g.latEdge)

                print(("Max in nVelocity: %e" % np.max(self.nVelocity)))
                print(("Max in u_true: %e" % np.max(u_true)))
                edgeInd = np.argmax(self.nVelocity)
                cell0 = g.cellsOnEdge[edgeInd, 0] - 1
                cell1 = g.cellsOnEdge[edgeInd, 1] - 1
                vertex0 = g.verticesOnEdge[edgeInd,0] - 1
                vertex1 = g.verticesOnEdge[edgeInd,1] - 1
                nVector = np.array([g.xCell[cell1] - g.xCell[cell0], g.yCell[cell1] - g.yCell[cell0], g.zCell[cell1] - g.zCell[cell0]])
                nVector /= np.sqrt(np.sum(nVector**2))
                hVector = np.array([-g.yEdge[edgeInd], g.xEdge[edgeInd], 0])
                hVector /= np.sqrt(np.sum(hVector**2))
                print(("latEdge[%d] = %e" % (edgeInd, g.latEdge[edgeInd]))) 
                print(("lonEdge[%d] = %e" % (edgeInd, g.lonEdge[edgeInd]))) 
                print(("Actual horizontal velocity at edge %d: %e" % (edgeInd, u_true[edgeInd])))
                print(("Actual normal velocity component: %e" % (u_true[edgeInd]*np.dot(nVector, hVector))))
                print(("Approximate normal velocity component: %e" % (self.nVelocity[edgeInd],)))
                print(("Actual psi at vertex %d: %e" % (vertex0, -a*u0*np.sin(g.latVertex[vertex0]) + a*u0*np.sin(g.latCell[0]))))
                print(("Approximate psi at vertex %d: %e" % (vertex0, self.psi_vertex[vertex0])))
                print(("Actual psi at vertex %d: %e" % (vertex1, -a*u0*np.sin(g.latVertex[vertex1]) + a*u0*np.sin(g.latCell[0]))))
                print(("Approximate psi at vertex %d: %e" % (vertex1, self.psi_vertex[vertex1])))
                print(("dvEdge[%d] = %e" % (edgeInd, g.dvEdge[edgeInd])))
                print("")


                print(("Max in tVelocity: %e" % np.max(self.tVelocity)))
                print(("Max in u_true: %e" % np.max(u_true)))
                print("")

                print(("Max in psi: %e" % np.max(self.psi_cell)))
                print(("Max in psi_vertex: %e" % np.max(self.psi_vertex)))
                print(("L-infinity error in psi: %e" % (np.max(np.abs(self.psi_cell - psi_true)) / np.max(np.abs(psi_true)),) ))
                print(("L-infinity error in psi_vertex: %e" % (np.max(np.abs(self.psi_vertex - psi_vertex_true)) / np.max(np.abs(psi_vertex_true)),) ))
                print("")

                print(("Max in phi: %e" % np.max(self.phi_cell)))
                print(("Max in phi_vertex: %e" % np.max(self.phi_vertex)))
                print("")

                raise ValueError("Abort after testing in start_from_function")

        elif c.test_case == 5:
            a = c.earth_radius
            u0 = 20.

            h0 = 5960.
            gh = c.gravity*h0 - np.sin(g.latCell[:])**2 * (a*c.Omega0*u0 + 0.5*u0*u0) 
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
            self.psi_cell[:] = -a * u0 * np.sin(g.latCell[:])
            self.psi_cell[:] -= self.psi_cell[0]
            self.phi_cell[:] = 0.
            self.psi_vertex[:] = -a * u0 * np.sin(g.latVertex[:])
            self.psi_vertex[:] -= self.psi_vertex[0]
            self.phi_vertex[:] = 0.
            
            #self.compute_diagnostics(g, c)

            self.curlWind_cell[:] = 0.
            self.divWind_cell[:] = 0.
            self.sfWind_cell[:] = 0.
            self.sfWind_vertex[:] = 0.
            self.vpWind_cell[:] = 0.
            self.vpWind_vertex[:] = 0.

        elif c.test_case == 11:
            # A wind-driven gyre at mid-latitude in the northern hemisphere
            tau0 = 1.e-4
            
            latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
            lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

            latmid = 0.5*(latmin+latmax)
            latwidth = latmax - latmin

            lonmid = 0.5*(lonmin+lonmax)
            lonwidth = lonmax - lonmin

            r = c.earth_radius

            self.vorticity[:] = 0.
            self.divergence[:] = 0.
            self.thickness[:] = 4000.

            self.psi_cell[:] = 0.0
            self.psi_vertex[:] = 0.0
            
            # Initialize wind
            self.curlWind_cell[:] = -tau0 * np.pi/(latwidth*r) * \
                                    np.sin(np.pi*(g.latCell[:]-latmin) / latwidth)
            self.divWind_cell[:] = 0.

        elif c.test_case == 12:
            # One gyre with no forcing, for a bounded domain over NA
            d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-1.1))**2/.3**2)
            f0 = np.mean(g.fCell)
            self.psi_cell[:] = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
            self.psi_cell[:] -= np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
            self.psi_cell *= c.gravity / f0
            self.vorticity = cmp.discrete_laplace( \
                 g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                 self.psi_cell)
            self.divergence[:] = 0.
            self.thickness[:] = 4000.
            
            # Initialize wind
            self.curlWind_cell[:] = 0.
            self.divWind_cell[:] = 0.

            # Eliminate bottom drag
            #c.bottomDrag = 0.

            # Eliminate lateral diffusion
            #c.delVisc = 0.
            #c.del2Visc = 0.
            
            
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
        c.dt = float(rdata.dt)
        c.delVisc = float(rdata.delVisc)
        c.bottomDrag = float(rdata.bottomDrag)
        c.on_sphere = bool(rdata.on_sphere)
        c.earth_radius = float(rdata.radius)
        c.f0 = float(rdata.f0)
        
        rdata.close( )

    def initialization(self, g, vc, c):

        if c.restart:
            self.restart_from_file(g,c)
        else:
            self.start_from_function(g, c)
            self.compute_diagnostics(g, vc, c)
            
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
        out.linear_solver = "%s" % (c.linear_solver)
        out.err_tol = "%e" % (c.err_tol)
        out.max_iter = "%d" % (c.max_iter)
        out.restart = "%s" % (c.restart)
        out.dt = "%f" % (c.dt)
        out.delVisc = "%e" % (c.delVisc)
        out.bottomDrag = "%e" % (c.bottomDrag)
        out.on_a_global_sphere = "%s" % (c.on_a_global_sphere)
        out.radius = "%e" % (c.earth_radius)
        out.no_flux_BC = "%s" % (c.no_flux_BC)
        out.no_slip_BC = "%s" % (c.no_slip_BC)
        
        out.close( )


    def compute_tendencies(self, g, c):

        thicknessTransport = self.thickness_edge[:] * self.nVelocity[:]
        self.tend_thickness[:] = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, thicknessTransport)
        
        absVorTransport = self.eta_edge[:] * self.nVelocity[:]
        self.tend_vorticity[:] = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorTransport)
        self.tend_vorticity[:] += self.curlWind_cell / self.thickness[:]
        self.tend_vorticity[:] -= c.bottomDrag * self.vorticity[:]
        self.tend_vorticity[:] += c.delVisc * cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, self.vorticity)
        
        absVorCirc = self.eta_edge[:] * self.tVelocity[:]
        geoPotent = c.gravity * (self.thickness[:] + g.bottomTopographyCell[:])  + self.kinetic_energy[:]
        self.tend_divergence[:] = cmp.discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorCirc) - \
                          cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, geoPotent)
        self.tend_divergence[:] += self.divWind_cell/self.thickness[:]
        self.tend_divergence[:] -= c.bottomDrag * self.divergence[:]
        self.tend_divergence[:] += c.delVisc * cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, self.divergence)

    def compute_diagnostics(self, g, vc, c):
        # Compute diagnostic variables from pv_cell

        if c.test_case == 1:
            #For shallow water test case #1, reset the vorticity and divergence to the initial states
            a = c.earth_radius
            u0 = 2*np.pi*a / (12*86400)
            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.

        self.thickness_vertex[:] = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.thickness)
        
        self.compute_psi_cell(vc, c)
        self.compute_phi_cell(vc, c)

        # Only to recalculate vorticity on the boundary to ensure zero average. Necessary for a global domain, or a bounded domain with no-slip BCs
        if c.on_a_global_sphere or c.no_slip_BC or c.delVisc > np.finfo('float32').tiny:
            self.vorticity[:] = cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, self.psi_cell)

        # Only to re-calcualte divergence[0] to ensure zero average. This is absoutely necessary when there are external forcings
        self.divergence[:] = cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, self.phi_cell)
            
        # Map vorticity and divergence to the dual mesh, and then compute the streamfunction and velocity potential
        # on dual mesh
        self.vorticity_vertex[:] = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.vorticity)
        self.divergence_vertex[:] = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, self.divergence)

        # Compute psi_vertex and phi_vertex from vorticity_vertex and divergence_vertex
        self.compute_psi_vertex(vc, c)
        self.compute_phi_vertex(vc, c)
        
        # compute the normal and tangential velocity components
        self.nVelocity = cmp.compute_normal_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_cell, self.psi_vertex)
        self.tVelocity = cmp.compute_tangential_velocity(g.verticesOnEdge, g.cellsOnEdge, g.dcEdge, g.dvEdge, self.phi_vertex, self.psi_cell)

        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness
        
        # Map from cell to edge
        self.pv_edge[:] = cmp.cell2edge(g.cellsOnEdge, self.pv_cell)
        self.thickness_edge[:] = cmp.cell2edge(g.cellsOnEdge, self.thickness)

        # Compute absolute vorticity on edge
        self.eta_edge[:] = self.pv_edge[:] * self.thickness_edge[:]

        # Compute kinetic energy
        #self.compute_kinetic_energy(g, c)
        kenergy_edge = 0.5 * (self.nVelocity * self.nVelocity + self.tVelocity * self.tVelocity )
        self.kinetic_energy[:] = cmp.edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, kenergy_edge)


        
    def compute_psi_cell(self, vc, c):
        # To compute the psi_cell using the elliptic equation on the
        # interior cells

        if not c.on_a_global_sphere:
            # A bounded domain
             vc.invLaplace_prime_dirich(self.vorticity, self.psi_cell)
        else:
            # A global domain with no boundary
            vc.invLaplace_prime_neumann(self.vorticity, self.psi_cell)
        return 0


    def compute_psi_vertex(self, vc, c):
        # To compute the psi_cell using the elliptic equation on the
        # interior cells

        if not c.on_a_global_sphere:
            vc.invLaplace_dual_dirich(self.vorticity_vertex, self.psi_vertex)
        else:
            # A global domain with no boundary
            vc.invLaplace_dual_neumann(self.vorticity_vertex, self.psi_vertex)
            
        return 0
    
    
    def compute_phi_cell(self, vc, c):
        # To compute the phi_cell from divergence

        vc.invLaplace_prime_neumann(self.divergence, self.phi_cell)
        return 0

    def compute_phi_vertex(self, vc, c):
        # To compute the phi_cell from divergence

        vc.invLaplace_dual_neumann(self.divergence_vertex, self.phi_vertex)
        return 0
    

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

        #self.compute_kinetic_energy(g, c)
        out.variables['kinetic_energy'][k,:,0]= self.kinetic_energy[:]
        
        out.close( )
        
#def iterative_solver(A, b, x0, c):
#    x, err = cg(A, b, x0=x0, tol=c.err_tol, maxiter=c.max_iter)#

#    if err > 0:
#        raise ValueError("Convergence not achieved after %d iterations." % err)
#    elif err < 0:
#        raise ValueError("Something is wrong in iterative_solver. Program abort.")
#    else:
#        return x

#def pysparse_iterative_solver(A, b, x, K, c):#

#    info, iter, relres = krylov.gmres(A, b, x, c.err_tol, c.max_iter, K)

#    if info < 0:
#        raise ValueError("Convergence not achieved after %d iterations. Error code %d." % (iter, info))
#    else:
        #print("x[1001] = %e" % x[1001])
#        print(("iter = %d" % iter))
#        print(("relres = %e" % relres))
#        return info
    
def timestepping_rk4_z_hex(s, s_pre, s_old, g, vc, c):

    coef = np.array([0., .5, .5, 1.])
    accum = np.array([1./6, 1./3, 1./3, 1./6])

    dt = c.dt

    s_intm = deepcopy(s_pre)

    # Update the time stamp first
    s.time += dt

    s.thickness[:] = s_pre.thickness[:]
    s.vorticity[:] = s_pre.vorticity[:]
    s.divergence[:] = s_pre.divergence[:]
    for i in range(4):

        # Compute the tendencies
        s_intm.compute_tendencies(g, c)

        # Accumulating the change in s
        s.thickness[:] += s_intm.tend_thickness[:]*accum[i]*dt
        s.vorticity[:] += s_intm.tend_vorticity[:]*accum[i]*dt
        s.divergence[:] += s_intm.tend_divergence[:]*accum[i]*dt

        ## Examine pot-enstrophy conservation
        #q2 = s_intm.pv_cell * s_intm.pv_cell
        #grad_q2 = cmp.discrete_grad_n(q2, g.cellsOnEdge, g.dcEdge)
        #grad_q = cmp.discrete_grad_n(s_intm.pv_cell, g.cellsOnEdge, g.dcEdge)
        #dArea = g.dcEdge * g.dvEdge * 0.5
        #pEns_tend = np.sum(grad_q2 * s_intm.thickness_edge * s_intm.nVelocity * dArea)
        #pEns_tend -= 2*np.sum(grad_q * s_intm.eta_edge * s_intm.nVelocity * dArea)
        #print("pEns_tend = %e" % pEns_tend)

        if i < 3:
            # Advance s_intm 
            s_intm.thickness[:] = s_pre.thickness[:] + coef[i+1]*dt*s_intm.tend_thickness[:]
            s_intm.vorticity[:] = s_pre.vorticity[:] + coef[i+1]*dt*s_intm.tend_vorticity[:]
            s_intm.divergence[:] = s_pre.divergence[:] + coef[i+1]*dt*s_intm.tend_divergence[:]

            if i == 0:
                s_intm.psi_cell[:] = 1.5*s_pre.psi_cell[:] - 0.5*s_old.psi_cell[:]
                s_intm.phi_cell[:] = 1.5*s_pre.phi_cell[:] - 0.5*s_old.phi_cell[:]
                s_intm.psi_vertex[:] = 1.5*s_pre.psi_vertex[:] - 0.5*s_old.psi_vertex[:]
                s_intm.phi_vertex[:] = 1.5*s_pre.phi_vertex[:] - 0.5*s_old.phi_vertex[:]

            if i==2:
                s_intm.psi_cell[:] = 2*s_intm.psi_cell[:] - s_pre.psi_cell[:]
                s_intm.phi_cell[:] = 2*s_intm.phi_cell[:] - s_pre.phi_cell[:]
                s_intm.psi_vertex[:] = 2*s_intm.psi_vertex[:] - s_pre.psi_vertex[:]
                s_intm.phi_vertex[:] = 2*s_intm.phi_vertex[:] - s_pre.phi_vertex[:]

            s_intm.compute_diagnostics(g, vc, c)

    # Prediction using the latest s_intm values
    s.psi_cell[:] = s_intm.psi_cell[:]
    s.phi_cell[:] = s_intm.phi_cell[:]
    s.psi_vertex[:] = s_intm.psi_vertex[:]
    s.phi_vertex[:] = s_intm.phi_vertex[:]
    
    s.compute_diagnostics(g, vc, c)


def timestepping_euler(s, g, c):

    dt = c.dt

    s_pre = deepcopy(s)

    # Update the time stamp first
    s.time += dt

    # Compute the tendencies
    thicknessTransport = s_pre.thickness_edge[:] * s_pre.nVelocity[:]
    s.tend_thickness = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, thicknessTransport)

    absVorTransport = s_pre.eta_edge[:] * s_pre.nVelocity[:]
    s.tend_vorticity = -cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorTransport)
    s.tend_vorticity += s.curlWind_cell / s_pre.thickness[:]
    s.tend_vorticity -= c.bottomDrag * s_pre.vorticity[:]
    s.tend_vorticity += c.delVisc * cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, s_pre.vorticity)

    absVorCirc = s_pre.eta_edge[:] * s_pre.tVelocity[:]
    geoPotent = c.gravity * (s_pre.thickness[:] + g.bottomTopographyCell[:])  + s_pre.kinetic_energy[:]
    s.tend_divergence = cmp.discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorCirc) - \
                      cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, geoPotent)
    s.tend_divergence += s.divWind_cell/s_pre.thickness[:]
    s.tend_divergence -= c.bottomDrag * s_pre.divergence[:]
    s.tend_divergence += c.delVisc * cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, s_pre.divergence)

    # Accumulating the change in s
    s.thickness[:] += s.tend_thickness[:]*dt
    s.vorticity[:] += s.tend_vorticity[:]*dt
    s.divergence[:] += s.tend_divergence[:]*dt


    s.compute_diagnostics(g, vc, c)

    
        
def main( ):


    # -----------------------------------------------------------
    # Create a grid_data object, a state_data object, and a parameter object.
    # -----------------------------------------------------------

    c = parameters()
    g = grid_data('grid.nc', c)
    vc = VectorCalculus(g, c)
    s = state_data(g, c)

#    from Testing import run_tests
#    run_tests(g, vc, c, s)
#    raise ValueError("Just for testing.")

    s.initialization(g, vc, c)
#    raise ValueError("Just for testing.")
    
    s_init = deepcopy(s)
    h0 = np.mean(s_init.thickness[:])
        
    # Compute energy and enstrophy
    kenergy = np.zeros(c.nTimeSteps+1)
    penergy = np.zeros(c.nTimeSteps+1)
    total_energy = np.zeros(c.nTimeSteps+1)
    mass = np.zeros(c.nTimeSteps+1)
    penstrophy = np.zeros(c.nTimeSteps+1)
    pv_max = np.zeros(c.nTimeSteps+1)
    pv_min = np.zeros(c.nTimeSteps+1)

    kenergy[0] = np.sum(s.thickness[:]*s.kinetic_energy[:]*g.areaCell[:])
    penergy[0] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
    total_energy[0] = kenergy[0] + penergy[0]
    mass[0] = np.sum(s.thickness[:] * g.areaCell[:])
    penstrophy[0] = 0.5 * np.sum(g.areaCell[:] * s.thickness * s.pv_cell[:]**2)
    pv_max[0] = np.max(s.pv_cell)
    pv_min[0] = np.min(s.pv_cell)


    print(("Running test case \#%d" % c.test_case))
    print(("K-nergy, p-energy, t-energy, p-enstrophy, mass: %e, %e, %e, %e, %e" % (kenergy[0], penergy[0], total_energy[0], penstrophy[0], mass[0])))

    error1 = np.zeros((c.nTimeSteps+1, 3)); error1[0,:] = 0.
    error2 = np.zeros((c.nTimeSteps+1, 3)); error2[0,:] = 0.
    errorInf = np.zeros((c.nTimeSteps+1, 3)); errorInf[0,:] = 0.

    s.save(c, g, 0)

    # Entering the loop
    t0 = time.clock( )
    t0a = time.time( )
    s_pre = deepcopy(s)
    s_old = deepcopy(s)
    
    for iStep in range(c.nTimeSteps):

        print(("Doing step %d " % iStep))

        if c.timestepping == 'RK4':
            timestepping_rk4_z_hex(s, s_pre, s_old, g, vc, c)
        elif c.timestepping == 'E':
            timestepping_euler(s, g, vc, c)
        else:
            raise ValueError("Invalid choice for time stepping")

        # Compute energy and enstrophy
        kenergy[iStep+1] = np.sum(s.thickness[:]*s.kinetic_energy[:]*g.areaCell[:])
        penergy[iStep+1] = 0.5*c.gravity* np.sum((s.thickness[:]-h0)**2 * g.areaCell[:])
        total_energy[iStep+1] = kenergy[iStep+1] + penergy[iStep+1]
        mass[iStep+1] = np.sum(s.thickness[:] * g.areaCell[:])
        penstrophy[iStep+1] = 0.5 * np.sum(g.areaCell[:] * s.thickness[:] * s.pv_cell[:]**2)
        pv_max[iStep+1] = np.max(s.pv_cell)
        pv_min[iStep+1] = np.min(s.pv_cell)
#        aVorticity_total[iStep+1] = np.sum(g.areaCell * s.eta[:])
        
        print(("K-nergy, p-energy, t-energy, p-enstrophy, mass: %.15e, %.15e, %.15e, %.15e, %.15e" % \
              (kenergy[iStep+1], penergy[iStep+1], total_energy[iStep+1], penstrophy[iStep+1], mass[iStep+1])))

        if kenergy[iStep+1] != kenergy[iStep+1]:
            raise ValueError("Exceptions detected in energy. Stop now")
        
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

        s_tmp = s_old
        s_old = s_pre
        s_pre = s
        s = s_tmp

    days = c.dt * np.arange(c.nTimeSteps+1) / 86400.
    t1 = time.clock( )
    t1a = time.time( )
    plt.close('all')

    plt.figure(0)
    plt.plot(days, kenergy, '--', label="Kinetic energy", hold=True)
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    #plt.ylim(2.5e17, 2.6e17)
    plt.legend(loc=1)
    plt.savefig('energy.png', format='PNG')

    plt.figure(6)
    plt.plot(days, penergy, '-.', label="Potential energy", hold=True)
    plt.plot(days, total_energy, '-', label="Total energy")
    plt.xlabel('Time (days)')
    plt.ylabel('Energy')
    #plt.ylim(8.0e20,8.15e20)
    plt.legend(loc=1)
    plt.savefig('total-energy.png', format='PNG')
    
    plt.figure(1)
    plt.plot(days, penstrophy)
    plt.xlabel('Time (days)')
    plt.ylabel('Enstrophy')
    #plt.ylim(0.74, 0.78)
    plt.savefig('enstrophy.png', format='PNG')
    print(("Change in potential enstrophy = %.15e " % (penstrophy[-1] - penstrophy[0])))

    plt.figure(5)
    plt.plot(days, mass)
    plt.xlabel('Time (days)')
    plt.ylabel('Mass')
    #plt.ylim(1.175e18, 1.225e18)
    plt.savefig('mass.png', format='PNG')

    plt.figure(7)
    plt.plot(days, pv_max, '-.', hold=True, label='PV max')
    plt.plot(days, pv_min, '--', hold=True, label='PV min')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.ylim([-0.0001, 0.0001])
    plt.xlabel('Days')
    plt.ylabel('Max/Min potential vorticity')
    plt.legend(loc=1)
    plt.savefig('pv_max_min.png', format='PNG')
    plt.savefig('pv_max_min.pdf', format='PDF')

    #plt.figure(8)
    #plt.plot(Years, aVorticity_total)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ##plt.ylim([-0.0001, 0.0001])
    #plt.xlabel('Years')
    #plt.ylabel('Total absolute vorticity')
    #plt.savefig('aVort_total.png', format='PNG')
    #plt.savefig('aVort_total.pdf', format='PDF')
    
    if c.test_case == 2:
        plt.figure(2); 
        plt.plot(days, error1[:,0], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,0], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,0], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-h.png', format='PNG')

        plt.figure(3); 
        plt.plot(days, error1[:,1], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,1], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,1], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Relative error')
        plt.savefig('error-vorticity.png', format='PNG')

        plt.figure(4); 
        plt.plot(days, error1[:,2], '--', label=r'$L^1$ norm', hold=True)
        plt.plot(days, error2[:,2], '-', label=r'$L^2$ norm')
        plt.plot(days, errorInf[:,2], '-.', label=r'$L^\infty$ norm')
        plt.legend(loc=1)
        plt.xlabel('Time (days)')
        plt.ylabel('Absolute error')
        plt.savefig('error-divergence.png', format='PNG')

        print("Final l2 errors for thickness, vorticity, and divergence:")
        print(("                    %e,        %e,     %e" % (error2[-1,0], error2[-1,1], error2[-1,2])))
        

    print(('CPU time used: %f seconds' % (t1-t0)))
    print(('Walltime used: %f seconds' % (t1a-t0a)))

        
if __name__ == '__main__':
    main( )


            
        
    
