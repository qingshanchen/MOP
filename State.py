import numpy as np
import Parameters as c
from Grid import grid_data
from ComputeEnvironment import ComputeEnvironment
from VectorCalculus import VectorCalculus
import netCDF4 as nc
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
#        self.kenergy = np.zeros(g.nCells)
        self.kenergy_edge = np.zeros(g.nEdges)
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

            
    def start_from_function(self, vc, g, c):

        latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
        lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

        latmid = 0.5*(latmin+latmax)
        latwidth = latmax - latmin

        lonmid = 0.5*(lonmin+lonmax)
        lonwidth = lonmax - lonmin

        pi = np.pi; sin = np.sin; exp = np.exp
        r = c.sphere_radius

        if c.test_case == 1:
            a = c.sphere_radius
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
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = np.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
            self.thickness[:] = gh / c.gravity
            h0 = gh0 / c.gravity

            ###
            #print("gh[10258] = %e" % gh[10258])
            #print("h[10258] = %e" % self.thickness[10258])
            #print("g*h[10258] = %e" % (c.gravity*self.thickness[10258]))
            #print("g = %f" % c.gravity)
            ###

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            
            self.psi_cell[:] = -a * h0 * u0 * np.sin(g.latCell[:]) 
            self.psi_cell[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
            self.psi_cell -= self.psi_cell[0]
            self.phi_cell[:] = 0.

            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)


            ## For debugging ##
            if False:
                # To check the consistency between psi_cell and vorticity
                self.thickness_edge = vc.cell2edge(self.thickness)
                self.psi_vertex[:] = vc.cell2vertex(self.psi_cell)
                nVelocity = vc.discrete_grad_n(self.phi_cell)
                nVelocity -= vc.discrete_grad_td(self.psi_vertex)
                nVelocity /= self.thickness_edge
                vorticity1 = vc.vertex2cell(vc.discrete_curl_t(nVelocity))
                err = vorticity1 - self.vorticity
                print("vorticity computed using normal vel.")
                print("relative error = %e" % (np.sqrt(np.sum(err**2*g.areaCell)/np.sum(self.vorticity**2*g.areaCell))))

                self.phi_vertex[:] = vc.cell2vertex(self.phi_cell)
                tVelocity = vc.discrete_grad_n(self.psi_cell)
                tVelocity += vc.discrete_grad_tn(self.phi_vertex)
                tVelocity /= self.thickness_edge
                vorticity2 = vc.discrete_curl_v(tVelocity)
                err = vorticity2 - self.vorticity
                print("vorticity computed using tang vel.")
                print("relative error = %e" % (np.sqrt(np.sum(err**2*g.areaCell)/np.sum(self.vorticity**2*g.areaCell))))

                err = 0.5*(vorticity1 + vorticity2) - self.vorticity
                print("vorticity computed using both normal and tang vel.")
                print("relative error = %e" % (np.sqrt(np.sum(err**2*g.areaCell)/np.sum(self.vorticity**2*g.areaCell))))
                raise ValueError("Testing the consistency between streamfunction and vorticity.")
                ## End of debugging ##
            
        elif c.test_case == 5:
            a = c.sphere_radius
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
            
            self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)

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

            r = c.sphere_radius

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
            self.thickness[:] = 4000.
            self.psi_cell[:] = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
#            self.psi_cell[:] -= np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
            self.psi_cell *= c.gravity / f0 * self.thickness
            self.phi_cell[:] = 0.
            self.vorticity = vc.discrete_laplace_v(self.psi_cell)
            self.vorticity /= self.thickness
            self.divergence[:] = 0.
            
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

        self.thickness[:] = rdata.variables['thickness'][start_ind,:,0]
        self.vorticity[:] = rdata.variables['vorticity_cell'][start_ind,:,0]
        self.divergence[:] = rdata.variables['divergence'][start_ind,:,0]
        self.psi_cell[:] = rdata.variables['psi_cell'][start_ind,:,0]
        self.phi_cell[:] = rdata.variables['phi_cell'][start_ind,:,0]
        self.time = rdata.variables['xtime'][start_ind]

        g.bottomTopographyCell[:] = rdata.variables['bottomTopographyCell'][:]
        self.SS0 = np.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / np.sum(g.areaCell)
        
        # Read simulation parameters
        c.test_case = int(rdata.test_case)
        c.dt = float(rdata.dt)
        c.delVisc = float(rdata.delVisc)
        c.bottomDrag = float(rdata.bottomDrag)
        c.sphere_radius = float(rdata.sphere_radius)
        
        rdata.close( )

    def initialization(self, g, vc, c):

        if c.do_restart:
            self.restart_from_file(g,c)
        else:
            self.start_from_function(vc, g, c)

        # Compute diagnostic variables
        self.compute_diagnostics(g, vc, c)
            
        # Open the output file and create new state variables
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        out.createVariable('xtime', 'f8', ('Time',))
        out.createVariable('thickness', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('vorticity_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('divergence', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('psi_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('phi_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('nVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('tVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
#        out.createVariable('kenergy', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('curlWind_cell', 'f8', ('nCells',))
        out.createVariable('bottomTopographyCell', 'f8', ('nCells',))

        # Record parameters used for this simulation
        out.test_case = "%d" % (c.test_case)
        out.conserve_enstrophy = "%s" % (c.conserve_enstrophy)
        out.timestepping = "%s" % (c.timestepping)
        out.linear_solver = "%s" % (c.linear_solver)
        if c.linear_solver == 'amgx':
            out.err_tol = "%e" % (c.err_tol)
            out.max_iters = "%d" % (c.max_iters)
        out.do_restart = "%s" % (c.do_restart)
        out.dt = "%f" % (c.dt)
        out.delVisc = "%e" % (c.delVisc)
        out.bottomDrag = "%e" % (c.bottomDrag)
        out.on_a_global_sphere = "%s" % (c.on_a_global_sphere)
        out.sphere_radius = "%e" % (c.sphere_radius)
        out.no_flux_BC = "%s" % (c.no_flux_BC)
        out.no_slip_BC = "%s" % (c.no_slip_BC)
        out.free_slip_BC = "%s" % (c.free_slip_BC)
        
        out.close( )


    def compute_tendencies(self, g, c, vc):

        # Tendency for thicknetss
        if c.component_for_hamiltonian == 'normal_tangent':
            self.vVertex[:] = vc.discrete_laplace_t(self.phi_vertex)
            self.tend_thickness[:] = -0.5 * vc.vertex2cell(self.vVertex)
            self.tend_thickness[:] += -0.5 * vc.discrete_laplace_v(self.phi_cell)
        else:
            self.tend_thickness[:] = -vc.discrete_laplace_v(self.phi_cell)

        # Tendency for vorticity
        if c.conserve_enstrophy:
            if c.component_for_hamiltonian in ['normal', 'tangential', 'mix']:
                pv_vertex = vc.cell2vertex(self.pv_cell)
                psi_edge = vc.cell2edge(self.psi_cell)

                self.vEdge[:] = vc.discrete_grad_n(self.psi_cell)
                self.vEdge *= self.pv_edge
                self.vEdge -= psi_edge * vc.discrete_grad_n(self.pv_cell)
                self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
                self.tend_vorticity[:] = 1./6 * vc.vertex2cell(self.vVertex)

                self.vEdge[:] = psi_edge * vc.discrete_skewgrad_n(pv_vertex)  # valid on a globe
                self.vEdge[:] -= self.pv_edge * vc.discrete_skewgrad_n(self.psi_vertex)
                self.tend_vorticity += 1./6 * vc.discrete_div_v(self.vEdge)

                self.vEdge[:] = vc.discrete_skewgrad_n(pv_vertex) * vc.discrete_grad_n(self.psi_cell)
                self.vEdge -= vc.discrete_skewgrad_n(self.psi_vertex) * vc.discrete_grad_n(self.pv_cell)
                self.tend_vorticity += 1./3 * vc.edge2cell(self.vEdge)

                ### Debugging ###
#                print("Contribution of { }_3zeta to enstrophy: %e" % np.sum(self.tend_vorticity * self.pv_cell * g.areaCell))
#                print("Contribution of { }_3zeta to energy: %e" % (-1*np.sum(self.tend_vorticity * self.psi_cell * g.areaCell)))
                ### End of Debugging ###

                self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
                self.tend_vorticity[:] -= vc.discrete_div_v(self.vEdge)

            else:
                raise ValueError("Invalid choice for Hamiltonian component.")
        
        else:
            self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
            self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
            self.tend_vorticity[:] = 0.5 * vc.vertex2cell(self.vVertex)

            self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad_n(self.psi_vertex)
            self.tend_vorticity[:] -= 0.5 * vc.discrete_div_v(self.vEdge)

            ### Debugging ###
#            print("Contribution of { }_2zeta to enstrophy: %e" % np.sum(self.tend_vorticity * self.pv_cell * g.areaCell))
#            print("Contribution of { }_2zeta to energy: %e" % (-1*np.sum(self.tend_vorticity * self.psi_cell * g.areaCell)))
            ### End of Debugging ###
                
            if c.component_for_hamiltonian == 'normal_tangent':
                self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
                self.tend_vorticity[:] -= 0.5 * vc.discrete_div_v(self.vEdge)

                self.vEdge[:] = self.pv_edge * vc.discrete_grad_tn(self.phi_vertex)
                self.vVertex[:] = vc.discrete_div_t(self.vEdge)
                self.tend_vorticity[:] -= 0.5 * vc.vertex2cell(self.vVertex)

            elif c.component_for_hamiltonian in ['normal', 'tangential', 'mix']:
                self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
                self.tend_vorticity[:] -= vc.discrete_div_v(self.vEdge)

            else:
                raise ValueError("Invalid value of component_for_hamiltonian")
        
        self.tend_vorticity[:] += self.curlWind_cell / self.thickness[:]
        self.tend_vorticity[:] -= c.bottomDrag * self.vorticity[:]
        self.tend_vorticity[:] += c.delVisc * vc.discrete_laplace_v(self.vorticity)

        # Tendency for divergence
        if c.component_for_hamiltonian == 'normal_tangent':
            self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
            self.tend_divergence[:] = 0.5 * vc.discrete_div_v(self.vEdge)
            
#            tend_divergence_1 = 0.5 * vc.discrete_div_v(self.vEdge)

            self.vEdge[:] = self.pv_edge * vc.discrete_grad_td(self.psi_vertex)
            self.vVertex[:] = vc.discrete_div_t(self.vEdge)
            self.tend_divergence[:] += 0.5 * vc.vertex2cell(self.vVertex)
            
#            tend_divergence_1[:] += 0.5 * vc.vertex2cell(self.vVertex)
            
        elif c.component_for_hamiltonian in ['normal', 'tangential', 'mix']:
            self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
            self.tend_divergence[:] = vc.discrete_div_v(self.vEdge)

#            tend_divergence_1 = vc.discrete_div_v(self.vEdge)
            
        else:
            raise ValueError("Invalid value of component_for_hamiltonian")

        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
        self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
        self.tend_divergence[:] += 0.5 * vc.vertex2cell(self.vVertex)

#        tend_divergence_2 = 0.5 * vc.vertex2cell(self.vVertex)
        
#        self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad_n(self.phi_vertex)
        self.vEdge[:] = cmp.discrete_skewgrad_nnat(self.phi_vertex, self.phi_cell, g.verticesOnEdge, g.cellsOnEdge, \
                                                   g.dvEdge)
        self.vEdge *= self.pv_edge
        self.tend_divergence[:] -= 0.5 * vc.discrete_div_v(self.vEdge)

#        tend_divergence_2[:] -= 0.5 * vc.discrete_div_v(self.vEdge)

        ## The boundary terms
        if not c.on_a_global_sphere:
            pv_bv_edge = 0.5*(self.pv_cell[vc.cellBoundary_ord[:-1]-1] + self.pv_cell[vc.cellBoundary_ord[1:]-1])
            phi_diff_edge = self.phi_cell[vc.cellBoundary_ord[1:]-1] - self.phi_cell[vc.cellBoundary_ord[:-1]-1]
            pv_phi_diff_edge = pv_bv_edge * phi_diff_edge
            self.tend_divergence[vc.cellBoundary_ord[0]-1] -= 1./4/g.areaCell[vc.cellBoundary_ord[0]-1] * \
                    (pv_phi_diff_edge[-1] + pv_phi_diff_edge[0])
            self.tend_divergence[vc.cellBoundary_ord[1:-1]-1] -= 1./4/g.areaCell[vc.cellBoundary_ord[1:-1]-1] * \
                    (pv_phi_diff_edge[:-1] + pv_phi_diff_edge[1:])

        if c.component_for_hamiltonian == 'normal_tangent':
            self.tend_divergence[:] -= 0.5 * vc.discrete_laplace_v(self.geoPot)

#            tend_divergence_3 = -0.5 * vc.discrete_laplace_v(self.geoPot)
            
            self.vVertex[:] = vc.cell2vertex(self.geoPot)
            self.vVertex[:] = vc.discrete_laplace_t(self.vVertex)
            self.tend_divergence[:] -= 0.5 * vc.vertex2cell(self.vVertex)

#            tend_divergence_3[:] -= 0.5 * vc.vertex2cell(self.vVertex)
            
        elif c.component_for_hamiltonian in ['normal', 'tangential', 'mix']:
            self.tend_divergence[:] -= vc.discrete_laplace_v(self.geoPot)

#            tend_divergence_3 = -vc.discrete_laplace_v(self.geoPot)
            
        else:
            raise ValueError("Invalid value of component_for_hamiltonian")

        
    def compute_diagnostics(self, g, vc, c):
        # Compute diagnostic variables from pv_cell

        if c.test_case == 1:
            #For shallow water test case #1, reset the vorticity and divergence to the initial states
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.

        self.thickness_edge[:] = vc.cell2edge(self.thickness)

        self.compute_psi_phi(vc, g, c)

        self.psi_vertex[:] = vc.cell2vertex(self.psi_cell)
        self.phi_vertex[:] = vc.cell2vertex(self.phi_cell)

        if c.on_a_global_sphere:
            pass
            # Reset value of vorticity and divergence at cell 0
            #self.vorticity[0] = -1 * np.sum(self.vorticity[1:]*g.areaCell[1:]) / g.areaCell[0]
            #self.divergence[0] = -1 * np.sum(self.divergence[1:]*g.areaCell[1:]) / g.areaCell[0]
            #print("Total vorticity = %e" % (np.sum(self.vorticity * g.areaCell)))
            #print("Total divergence = %e" % (np.sum(self.divergence * g.areaCell)))
        else:
#            raise ValueError
            pass
        
        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness
        
        # Map from cell to edge
        self.pv_edge[:] = vc.cell2edge(self.pv_cell)

        # Compute kinetic energy on the edge
        if c.component_for_hamiltonian == 'normal':
            self.compute_kenergy_edge_n(vc, g, c)
        elif c.component_for_hamiltonian == 'tangential':
            self.compute_kenergy_edge_t(vc, g, c)
        elif c.component_for_hamiltonian == 'normal_tangent':
            self.compute_kenergy_edge_nt(vc, g, c)
        elif c.component_for_hamiltonian == 'mix':
            self.compute_kenergy_edge_mix(vc, g, c)
        else:
            raise ValueError("Invalid value for component_for_hamiltonian")

        self.geoPot[:] = c.gravity * (self.thickness[:] + g.bottomTopographyCell[:])
        self.geoPot[:] += vc.edge2cell(self.kenergy_edge[:])

        # Compute kinetic energy, total energy, and potential enstrophy
        self.kinetic_energy = np.sum(self.kenergy_edge * self.thickness_edge * g.areaEdge)
        
        self.pot_energy = 0.5 * c.gravity * np.sum((self.thickness[:] + g.bottomTopographyCell - self.SS0)**2 * g.areaCell[:])
        self.pot_enstrophy = 0.5 * np.sum(g.areaCell[:] * self.thickness * self.pv_cell[:]**2)


    def compute_psi_phi(self, vc, g, c):
        # To compute the psi_cell and phi_cell

        # Update the coefficient matrix for the coupled system
        vc.update_matrix_for_coupled_elliptic(self.thickness_edge, c, g)
        
        self.vortdiv[:g.nCells] = self.vorticity * g.areaCell
        self.vortdiv[g.nCells:] = self.divergence * g.areaCell
        self.psiphi[:g.nCells] = self.psi_cell[:]
        self.psiphi[g.nCells:] = self.phi_cell[:]
        
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
        out.variables['psi_cell'][k,:,0] = self.psi_cell[:]
        out.variables['phi_cell'][k,:,0] = self.phi_cell[:]
        out.variables['nVelocity'][k,:,0] = self.nVelocity[:]
        out.variables['tVelocity'][k,:,0] = self.tVelocity[:]

        if k==0:
            out.variables['curlWind_cell'][:] = self.curlWind_cell[:]
            out.variables['bottomTopographyCell'][:] = g.bottomTopographyCell[:]
        
        out.close( )

    def compute_tc2_errors(self, iStep, s_init, error1, error2, errorInf, g):
        # For test case #2, compute the errors
        error1[iStep+1, 0] = np.sum(np.abs(self.thickness[:] - s_init.thickness[:])*g.areaCell[:]) / np.sum(np.abs(s_init.thickness[:])*g.areaCell[:])
        error1[iStep+1, 1] = np.sum(np.abs(self.vorticity[:] - s_init.vorticity[:])*g.areaCell[:]) / np.sum(np.abs(s_init.vorticity[:])*g.areaCell[:])
        error1[iStep+1, 2] = np.max(np.abs(self.divergence[:] - s_init.divergence[:])) 

        error2[iStep+1, 0] = np.sqrt(np.sum((self.thickness[:] - s_init.thickness[:])**2*g.areaCell[:]))
        error2[iStep+1,0] /= np.sqrt(np.sum((s_init.thickness[:])**2*g.areaCell[:]))
        error2[iStep+1, 1] = np.sqrt(np.sum((self.vorticity[:] - s_init.vorticity[:])**2*g.areaCell[:]))
        error2[iStep+1,1] /= np.sqrt(np.sum((s_init.vorticity[:])**2*g.areaCell[:]))
        error2[iStep+1, 2] = np.max(np.abs(self.divergence[:] - s_init.divergence[:])) 

        errorInf[iStep+1, 0] = np.max(np.abs(self.thickness[:] - s_init.thickness[:])) / np.max(np.abs(s_init.thickness[:]))
        errorInf[iStep+1, 1] = np.max(np.abs(self.vorticity[:] - s_init.vorticity[:])) / np.max(np.abs(s_init.vorticity[:]))
        errorInf[iStep+1, 2] = np.max(np.abs(self.divergence[:] - s_init.divergence[:]))


    def compute_kenergy_edge_t(self, vc, g, c):
        # Compute the kinetic energy
        self.tVelocity[:] = vc.discrete_skewgrad_t(self.psi_cell)
        self.tVelocity += vc.discrete_grad_tn(self.phi_vertex)
        self.tVelocity /= self.thickness_edge

        self.kenergy_edge[:] = self.tVelocity * self.tVelocity


    def compute_kenergy_edge_n(self, vc, g, c):
        # Compute the kinetic energy
        self.nVelocity[:] = vc.discrete_grad_n(self.phi_cell)
        self.nVelocity -= vc.discrete_grad_td(self.psi_vertex)
        self.nVelocity /= self.thickness_edge

        self.kenergy_edge[:] = self.nVelocity * self.nVelocity


    def compute_kenergy_edge_nt(self, vc, g, c):
        # Compute the kinetic energy
        self.nVelocity[:] = vc.discrete_grad_n(self.phi_cell)
        self.nVelocity -= vc.discrete_grad_td(self.psi_vertex)
        self.nVelocity /= self.thickness_edge

        self.tVelocity[:] = vc.discrete_skewgrad_t(self.psi_cell)
        self.tVelocity += vc.discrete_grad_tn(self.phi_vertex)
        self.tVelocity /= self.thickness_edge
        
        self.kenergy_edge[:] = 0.5 * self.nVelocity * self.nVelocity
        self.kenergy_edge[:] += 0.5 * self.tVelocity * self.tVelocity

    def compute_kenergy_edge_mix(self, vc, g, c):
        # Compute the kinetic energy
        self.vEdge = vc.discrete_skewgrad_t(self.psi_cell)
        self.kenergy_edge[:] = self.vEdge**2

        self.vEdge = vc.discrete_grad_n(self.phi_cell)
        self.kenergy_edge += self.vEdge**2

        self.kenergy_edge += vc.discrete_skewgrad_n(self.psi_vertex) * vc.discrete_grad_n(self.phi_cell)
        self.kenergy_edge += vc.discrete_skewgrad_t(self.psi_cell) * vc.discrete_grad_tn(self.phi_vertex)

        self.kenergy_edge /= self.thickness_edge**2
        
        
def timestepping_rk4_z_hex(s, s_pre, s_old, s_old1, g, vc, c):

    coef = np.array([0., .5, .5, 1.])
    accum = np.array([1./6, 1./3, 1./3, 1./6])

    dt = c.dt

    s_intm = deepcopy(s_pre)

    # Update the time stamp first
    s.time = s_pre.time + dt

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

            if i == 1:
                pass
                
            if i==2:
                s_intm.psi_cell[:] = 2*s_intm.psi_cell[:] - s_pre.psi_cell[:]
                s_intm.phi_cell[:] = 2*s_intm.phi_cell[:] - s_pre.phi_cell[:]

            s_intm.compute_diagnostics(g, vc, c)

    # Prediction using the latest s_intm values
    s.psi_cell[:] = s_intm.psi_cell[:]
    s.phi_cell[:] = s_intm.phi_cell[:]
    
    s.compute_diagnostics(g, vc, c)

def timestepping_euler(s, g, c):

    dt = c.dt

    s_pre = deepcopy(s)

    # Update the time stamp first
    s.time += dt

    # Compute the tendencies
    thicknessTransport = s_pre.thickness_edge[:] * s_pre.nVelocity[:]
    s.tend_thickness = -cmp.discrete_div_v(g.cellsOnEdge, g.dvEdge, g.areaCell, thicknessTransport)

    absVorTransport = s_pre.eta_edge[:] * s_pre.nVelocity[:]
    s.tend_vorticity = -cmp.discrete_div_v(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorTransport)
    s.tend_vorticity += s.curlWind_cell / s_pre.thickness[:]
    s.tend_vorticity -= c.bottomDrag * s_pre.vorticity[:]
    s.tend_vorticity += c.delVisc * cmp.discrete_laplace_v(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, s_pre.vorticity)

    absVorCirc = s_pre.eta_edge[:] * s_pre.tVelocity[:]
    geoPotent = c.gravity * (s_pre.thickness[:] + g.bottomTopographyCell[:])  + s_pre.kenergy[:]
    s.tend_divergence = cmp.discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell, absVorCirc) - \
                      cmp.discrete_laplace_v(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, geoPotent)
    s.tend_divergence += s.divWind_cell/s_pre.thickness[:]
    s.tend_divergence -= c.bottomDrag * s_pre.divergence[:]
    s.tend_divergence += c.delVisc * cmp.discrete_laplace_v(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, s_pre.divergence)

    # Accumulating the change in s
    s.thickness[:] += s.tend_thickness[:]*dt
    s.vorticity[:] += s.tend_vorticity[:]*dt
    s.divergence[:] += s.tend_divergence[:]*dt


    s.compute_diagnostics(g, vc, c)

    
