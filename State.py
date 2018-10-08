import numpy as np
import Parameters as c
from Grid import grid_data
from ComputeEnvironment import ComputeEnvironment
from VectorCalculus import VectorCalculus
import netCDF4 as nc
#from matplotlib import use
#use('Agg')
#import matplotlib.pyplot as plt
from swe_comp import swe_comp as cmp
import os
from copy import deepcopy as deepcopy

max_int = np.iinfo('int32').max
max_double = np.finfo('float64').max


class state_data:
    def __init__(self, vc, g, c):

        # Prognostic variables
        self.thickness = np.zeros(g.nCells)
        self.vorticity = self.thickness.copy()
        self.divergence = self.thickness.copy()

        # Diagnostic variables
        self.thickness_vertex = np.zeros(g.nVertices)
        self.vorticity_vertex = np.zeros(g.nVertices)
        self.divergence_vertex = np.zeros(g.nVertices)
        self.vortdiv = np.zeros(2*g.nCells)

        self.psi_cell = np.zeros(g.nCells)
        self.psi_vertex = np.zeros(g.nVertices)
        self.psi_vertex_pred = np.zeros(g.nVertices)
        self.phi_cell = np.zeros(g.nCells)
        self.phi_vertex = np.zeros(g.nVertices)
        self.psiphi = np.zeros(2*g.nCells)
        
        self.nVelocity = np.zeros(g.nEdges)
        self.tVelocity = np.zeros(g.nEdges)
        self.pv_cell = np.zeros(g.nCells)
        self.pv_edge = np.zeros(g.nEdges)
        self.thickness_edge = np.zeros(g.nEdges)
        self.eta_cell = np.zeros(g.nCells)
        self.eta_edge = np.zeros(g.nEdges)
        self.kenergy = np.zeros(g.nCells)
        self.geoPot = np.zeros(g.nCells)

        self.SS0 = 0.     # Sea Surface at rest
        self.kinetic_energy = 0.
        self.pot_energy = 0.
        self.pot_enstrophy = 0.
        
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

        # Some generic temporary vectors
        self.vEdge = np.zeros(g.nEdges)
        self.vCell = np.zeros(g.nCells)
        self.vVertex = np.zeros(g.nVertices)
        
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

            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)
            

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

            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)

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
            
            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)

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
            
            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)

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
            
            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)
            
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
        out.createVariable('kenergy', 'f8', ('Time', 'nCells', 'nVertLevels'))
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
        out.free_slip_BC = "%s" % (c.free_slip_BC)
        
        out.close( )


    def compute_tendencies(self, g, c, vc):

        # Tendency for thicknetss 
        self.tend_thickness[:] = -vc.discrete_laplace(self.phi_cell)

        # Tendency for vorticity
        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
        self.vVertex[:] = vc.discrete_curl_trig(self.vEdge)
        self.tend_vorticity[:] = 0.5 * vc.vertex2cell(self.vVertex)

        self.vVertex[:] = vc.cell2vertex(self.psi_cell)
        self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad(self.vVertex)
        self.tend_vorticity[:] -= 0.5 * vc.discrete_div(self.vEdge)

        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
        self.tend_vorticity[:] -= vc.discrete_div(self.vEdge)
        
        self.tend_vorticity[:] += self.curlWind_cell / self.thickness[:]
        self.tend_vorticity[:] -= c.bottomDrag * self.vorticity[:]
        self.tend_vorticity[:] += c.delVisc * vc.discrete_laplace(self.vorticity)

        # Tendency for divergence
        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
        self.tend_divergence[:] = vc.discrete_div(self.vEdge)

        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
        self.vVertex[:] = vc.discrete_curl_trig(self.vEdge)
        self.tend_divergence[:] += 0.5 * vc.vertex2cell(self.vVertex)

        self.vVertex[:] = vc.cell2vertex(self.phi_cell)
        self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad(self.vVertex)
        self.tend_divergence[:] -= 0.5 * vc.discrete_div(self.vEdge)

        self.tend_divergence[:] -= vc.discrete_laplace(self.geoPot)
        
    def compute_diagnostics(self, g, vc, c):
        # Compute diagnostic variables from pv_cell

        if c.test_case == 1:
            #For shallow water test case #1, reset the vorticity and divergence to the initial states
            a = c.earth_radius
            u0 = 2*np.pi*a / (12*86400)
            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.

        self.thickness_edge[:] = vc.cell2edge(self.thickness)
        self.thickness_vertex[:] = vc.cell2vertex(self.thickness)

        self.compute_psi_phi(vc, g, c)

        if c.on_a_global_sphere:
            # Reset value of vorticity and divergence at cell 0
            self.vorticity[0] = -1 * np.sum(self.vorticity[1:]*g.areaCell[1:]) / g.areaCell[0]
            self.divergence[0] = -1 * np.sum(self.divergence[1:]*g.areaCell[1:]) / g.areaCell[0]
        else:
            raise ValueError
        
        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness
        
        # Map from cell to edge
        self.pv_edge[:] = vc.cell2edge(self.pv_cell)

        # Compute the kinetic energy
        self.nVelocity[:] = vc.discrete_grad_n(self.phi_cell)
        self.nVelocity -= vc.discrete_grad_td(self.psi_vertex)
        self.nVelocity /= self.thickness_edge

        self.kenergy[:] = vc.edge2cell(self.nVelocity * self.nVelocity)

        self.geoPot[:] = c.gravity * (self.thickness[:] + g.bottomTopographyCell[:])  + self.kenergy[:]

        # Compute kinetic energy, total energy, and potential enstrophy
        self.kinetic_energy = np.sum(self.nVelocity**2 * self.thickness_edge * g.areaEdge)
        self.pot_energy = 0.5 * c.gravity * np.sum((self.thickness[:] + g.bottomTopographyCell - self.SS0)**2 * g.areaCell[:])
        self.pot_enstrophy = 0.5 * np.sum(g.areaCell[:] * self.thickness * self.pv_cell[:]**2)


    def compute_psi_phi(self, vc, g, c):
        # To compute the psi_cell and phi_cell

        # Update the coefficient matrix for the coupled system
        vc.update_matrix_for_coupled_elliptic(self.thickness_edge, c, g)
        
        self.vortdiv[:g.nCells] = self.vorticity * g.areaCell
        self.vortdiv[g.nCells:] = self.divergence * g.areaCell
        
        if c.on_a_global_sphere:
            # A global domain with no boundary
            self.vortdiv[0] = 0.   # Set first element to zeor to make psi_cell[0] zero
            self.vortdiv[g.nCells] = 0.   # Set first element to zeor to make phi_cell[0] zero
            
        else:
            # A bounded domain with homogeneous Dirichlet for the psi and
            # homogeneous Neumann for phi
            self.vortdiv[vc.cellBoundary-1] = 0.   # Set boundary elements to zeor to make psi_cell zero there
            self.vortdiv[g.nCells] = 0.   # Set first element to zeor to make phi_cell[0] zero

        vc.POcpl.solve(vc.coefM, self.vortdiv, self.psiphi, linear_solver = c.linear_solver)
        self.psi_cell[:] = self.psiphi[:g.nCells]
        self.phi_cell[:] = self.psiphi[g.nCells:]


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

        #self.compute_kenergy(g, c)
        out.variables['kenergy'][k,:,0]= self.kenergy[:]
        
        out.close( )
        
    
def timestepping_rk4_z_hex(s, s_pre, s_old, s_old1, g, vc, c):

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
        s_intm.compute_tendencies(g, c, vc)

        # Accumulating the change in s
        s.thickness[:] += s_intm.tend_thickness[:]*accum[i]*dt
        s.vorticity[:] += s_intm.tend_vorticity[:]*accum[i]*dt
        s.divergence[:] += s_intm.tend_divergence[:]*accum[i]*dt

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

            if i == 1:
                pass
                
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
    geoPotent = c.gravity * (s_pre.thickness[:] + g.bottomTopographyCell[:])  + s_pre.kenergy[:]
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

    
