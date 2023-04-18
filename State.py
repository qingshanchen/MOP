import numpy as np
import netCDF4 as nc
from swe_comp import swe_comp as cmp
from copy import deepcopy as deepcopy

# Load appropriate module for working with objects on CPU / GPU
# (import here so xp is available in all methods below)
from Parameters import use_gpu
if use_gpu:
    import cupy as xp
else:
    import numpy as xp

class state_data:
    def __init__(self, vc, g, c):
            
        # Prognostic variables
        self.thickness = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.l1Thickness = xp.zeros(g.nCells)
        self.vorticity = self.thickness.copy()
        self.divergence = self.thickness.copy()

        # Diagnostic variables
        self.vorticity_vertex = xp.zeros(g.nVertices)
        self.divergence_vertex = xp.zeros(g.nVertices)
        self.circulation = xp.zeros(g.nCells)
        self.flux = xp.zeros(g.nCells)
        self.vortdiv = xp.zeros(2*g.nCells)

        self.psi_cell = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.psi_vertex = xp.zeros( (g.nVertices,c.nLayers), order=c.vector_order )
        self.psi_vertex_pred = xp.zeros(g.nVertices)
        self.phi_cell = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.phi_vertex = xp.zeros( (g.nVertices,c.nLayers), order=c.vector_order )
        self.psiphi = xp.zeros(2*g.nCells)
        
        self.nVelocity = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.tVelocity = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.pv_cell = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.pv_edge = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.thickness_edge = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.eta_cell = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.eta_edge = xp.zeros(g.nEdges)
        self.kenergy_edge = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.kenergy = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.geoPot = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )

        self.SS0 = xp.zeros(c.nLayers)     # Sea Surface at rest
        self.kinetic_energy = 0. #xp.zeros(c.nLayers)
        self.pot_energy = 0. #xp.zeros(c.nLayers)
        self.art_energy = 0. 
        self.pot_enstrophy = 0. #xp.zeros(c.nLayers)
        
        self.tend_thickness = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.tend_vorticity = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.tend_divergence = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )

        # Forcing
        self.curlWind_cell = xp.zeros(g.nCells)
        self.divWind_cell = xp.zeros(g.nCells)

        # Some generic temporary vectors
        self.vEdge = xp.zeros( (g.nEdges,c.nLayers), order=c.vector_order )
        self.vCell = xp.zeros( (g.nCells,c.nLayers), order=c.vector_order )
        self.vVertex = xp.zeros( (g.nVertices,c.nLayers), order=c.vector_order )
        
        # Time keeper
        self.time = 0.0

            
    def start_from_function(self, vc, g, c):

        latmin = xp.min(g.latCell[:]); latmax = xp.max(g.latCell[:])
        lonmin = xp.min(g.lonCell[:]); lonmax = xp.max(g.lonCell[:])

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
            r = a*xp.arccos(xp.sin(lat_c)*xp.sin(g.latCell[:]) + \
                xp.cos(lat_c)*xp.cos(g.latCell[:])*xp.cos(g.lonCell[:]-lon_c))
            self.thickness[:] = xp.where(r<=R, 0.25*h0*(1+xp.cos(np.pi*r/R)), 0.) + h0

            self.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
            self.divergence[:] = 0.
            #self.compute_diagnostics(g, c)

            self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / xp.sum(g.areaCell)
            

        elif c.test_case == 2:
            # SWSTC #2, with a stationary analytic solution 
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)   # Standard setup
            gh0 = 2.94e4
#            u0 = 2*np.pi*a / (6*86400)     # Increased/reduced max vel
#            gh0 = 50000.
            gh = xp.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0

            total_thickness = gh / c.gravity
            constant_layer_thickness = 400.

            if c.nLayers == 1:
                self.thickness[:,0] = total_thickness[:,0]
            else:
                ### Only one of these cases should be active at any time
                # Non-interactive case
    #            self.thickness[:] = total_thickness[:]

                # Interactive case: top layer variable thickness, others constant thickness
                self.thickness[:,1:] = constant_layer_thickness
                self.thickness[:,0] = total_thickness[:,0] - xp.sum(self.thickness[:,1:], axis=1)
    #            self.l1Thickness[:] = self.thickness[:,0]

                # Interactive case: bottom layer variable thickness, others constant thickness
    #            self.thickness[:,:-1] = constant_layer_thickness
    #            self.thickness[:,-1] = total_thickness[:,0] - xp.sum(self.thickness[:,:-1], axis=1)
    #            self.l1Thickness[:] = self.thickness[:,-1]
            
            h0 = gh0 / c.gravity
            
            self.vorticity[:] = 2*u0/a * xp.sin(g.latCell[:])
            self.divergence[:] = 0.

            if c.nLayers == 1:
                self.psi_cell[:,0] = -a * h0 * u0 * xp.sin(g.latCell[:,0]) 
                self.psi_cell[:,0] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (xp.sin(g.latCell[:,0]))**3 / 3.
                self.psi_cell[:,0] -= self.psi_cell[0,0]
            else:
                self.psi_cell[:,1:] = -a * u0 * constant_layer_thickness * xp.sin(g.latCell[:])
                self.psi_cell[:,0] = -a * h0 * u0 * xp.sin(g.latCell[:,0]) 
                self.psi_cell[:,0] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (xp.sin(g.latCell[:,0]))**3 / 3.
                self.psi_cell[:,0] += a*u0*(c.nLayers-1)*constant_layer_thickness * xp.sin(g.latCell[:,0])
                self.psi_cell[:,:] -= self.psi_cell[0,:]
            
            self.phi_cell[:,:] = 0.

            # Average thickness of each layer (which remains constant due to mass conservation)
            self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
            # Average topo height (static) 
            topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
            # Average height of layer interfaces
            for layer in range(c.nLayers):
                self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg

            print('Sea/layer sufrace average height:')
            print(self.SS0)
                                    
            
        elif c.test_case == 5:
            #SWSTC #5: zonal flow over a mountain topography
            
            a = c.sphere_radius
            u0 = 20.

            h0 = 5960.
            gh = c.gravity*h0 - xp.sin(g.latCell[:])**2 * (a*c.Omega0*u0 + 0.5*u0*u0) 
            h = gh / c.gravity

            # Define the mountain topography
            h_s0 = 2000.
            R = np.pi / 9
            lat_c = np.pi / 6.
            lon_c = -.5*np.pi
            r = xp.sqrt((g.latCell[:]-lat_c)**2 + (g.lonCell[:]-lon_c)**2)
            r = xp.where(r < R, r, R)
            g.bottomTopographyCell[:] = h_s0 * ( 1 - r/R)

            if c.nLayers == 1:
                self.thickness[:,0] = h[:,0] - g.bottomTopographyCell[:,0]
                
            elif c.nLayers == 2:
                ## Interactive case
                self.thickness[:,0] = h[:,0] - 3500.
                self.thickness[:,1] = 3500. - g.bottomTopographyCell[:,0]

            else:
                raise ValueError('This test case only takes nLayers = 1 or 2.')

            self.vorticity[:] = 2*u0/a * xp.sin(g.latCell[:])
            self.divergence[:] = 0.

            self.curlWind_cell[:] = 0.
            self.divWind_cell[:] = 0.
            
            self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
            topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
            for layer in range(c.nLayers):
                self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg

            print('Sea/layer sufrace average height:')
            print(self.SS0)

            
        elif c.test_case == 6:
            # Setup shallow water test case 6: Rossby-Haurwitz Wave
            #
            # Reference: Williamson, D.L., et al., "A Standard Test Set for Numerical 
            #            Approximations to the Shallow Water Equations in Spherical 
            #            Geometry" J. of Comp. Phys., 102, pp. 211--224

            a = c.sphere_radius
            h0 = 8000.
            w = 7.848e-6
            K = 7.848e-6
            R = 4

            cos_lat = xp.cos(g.latCell)
            A = 0.5*w*(2*c.Omega0 + w)*cos_lat**2 + \
                0.25*K**2*cos_lat**(2*R) * ( \
                (R+1)*cos_lat**2 + \
                (2*R**2 - R -2) - \
                2*R**2 / (cos_lat**2) )
            B = 2*(c.Omega0 + w)*K/(R+1)/(R+2)*cos_lat**R * \
                (R**2 + 2*R + 2 - (R+1)**2*cos_lat**2)
            C = 0.25*K**2*cos_lat**(2*R) * \
                ((R+1)*cos_lat**2 - (R+2))

            # The thickness field
            self.thickness[:] = c.gravity * h0 + a**2*A + \
                                a**2*B*xp.cos(R*g.lonCell) + \
                                a**2*C*xp.cos(2*R*g.lonCell)
            self.thickness[:] /= c.gravity

            # Vorticity and divergence fields
            self.vorticity[:] = 2*w*xp.sin(g.latCell) - \
                                K*xp.sin(g.latCell)*cos_lat**R* \
                                (R**2 + 3*R + 2)*xp.cos(R*g.lonCell)
            self.divergence[:] = 0.
                                                     
            self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / xp.sum(g.areaCell)


        elif c.test_case == 7:
            # Setup shallow water test case 7: Height and wind at 500 mb 
            #
            # Reference: Williamson, D.L., et al., "A Standard Test Set for Numerical 
            #            Approximations to the Shallow Water Equations in Spherical 
            #            Geometry" J. of Comp. Phys., 102, pp. 211--224
            ini_dat = np.loadtxt('tc7-init-on-%d.dat' % g.nCells)
            self.thickness[:] = ini_dat[:,2]
            self.vorticity[:] = ini_dat[:,3]
            self.divergence[:] = ini_dat[:,4]

            self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / xp.sum(g.areaCell)

        elif c.test_case == 8:
            # Setup shallow water test case 8: barotropic instability test case
            #
            # Reference:
            self.divergence[:] = 0.

            # Compute the vorticity field
            if c.use_gpu:
                self.vorticity[:] = xp.asarray(cmp.compute_swtc8_vort(g.latCell.get()))
            else:
                self.vorticity[:] = cmp.compute_swtc8_vort(g.latCell)

            # Compute the thickness field, and shift it towards 10km average depth
            if c.use_gpu:
                gh = xp.asarray(cmp.compute_swtc8_gh(g.latCell.get()))
            else:
                gh = cmp.compute_swtc8_gh(g.latCell)
                
            gh += 10000.*c.gravity - xp.sum(gh*g.areaCell)/xp.sum(g.areaCell)
            self.thickness[:] = gh / c.gravity

            # Add a small perturbation to the thickness field
            pert = 120*xp.cos(g.latCell)
            if xp.max(g.lonCell) > 1.1*np.pi:
                print("Found the range of lonCell to be [0, 2pi]")
                print("Shifting it to [-pi, pi]")
                g.lonCell[:] = xp.where(g.lonCell > np.pi, g.lonCell-2*np.pi, g.lonCell)
                g.lonVertex[:] = xp.where(g.lonVertex > np.pi, g.lonVertex-2*np.pi, g.lonVertex)
                g.lonEdge[:] = np.where(g.lonEdge > np.pi, g.lonEdge-2*np.pi, g.lonEdge)
            pert *= xp.exp(-(g.lonCell*3)**2)
            pert *= xp.exp(-((g.latCell-np.pi/4)*15)**2)
#            self.thickness[:] += pert

            self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / xp.sum(g.areaCell)
            
        elif c.test_case == 12:
            # SWSTC #2, with a stationary analytic solution, modified for the northern hemisphere
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            gh0 = 2.94e4
            gh = xp.sin(g.latCell[:])**2
            gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0

            total_thickness = gh / c.gravity
            constant_layer_thickness = 250.
            
            # Non-interactive case
#            self.thickness[:,:] = total_thickness[:,:]

            # Interactive case: top layer variable thickness, others constant thickness
            self.thickness[:,1:] = constant_layer_thickness
            self.thickness[:,0] = total_thickness[:,0] - xp.sum(self.thickness[:,1:], axis=1)
            if xp.any(self.thickness[:,0] < 0.):
                raise ValueError('Negative layer thickness detected during the initialization phase. Aborting.')


            h0 = gh0 / c.gravity
            
            self.vorticity[:] = 2*u0/a * xp.sin(g.latCell[:])
            self.divergence[:] = 0.

            self.psi_cell[:,1:] = -a * u0 * constant_layer_thickness * xp.sin(g.latCell[:])
            self.psi_cell[:,0] = -a * h0 * u0 * xp.sin(g.latCell[:,0]) 
            self.psi_cell[:,0] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (xp.sin(g.latCell[:,0]))**3 / 3.
            self.psi_cell[:,0] += a*u0*(c.nLayers-1)*constant_layer_thickness * xp.sin(g.latCell[:,0])
            self.phi_cell[:] = 0.

            self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
            topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
            for layer in range(c.nLayers):
                self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg

            print('Sea/layer sufrace average height:')
            print(self.SS0)


        elif c.test_case == 15:
            # Zonal flow over a mountain topography, on the northern hemisphere,
            # modeled after SWSTC #5
            
            a = c.sphere_radius
            u0 = 20.

            h0 = 5960.
            gh = c.gravity*h0 - xp.sin(g.latCell[:])**2 * (a*c.Omega0*u0 + 0.5*u0*u0) 
            h = gh / c.gravity

            # Define the mountain topography
            h_s0 = 2000.
            R = np.pi / 9
            lat_c = np.pi / 6.
            lon_c = .5*np.pi
            r = xp.sqrt((g.latCell[:]-lat_c)**2 + (g.lonCell[:]-lon_c)**2)
            r = xp.where(r < R, r, R)
            g.bottomTopographyCell[:] = h_s0 * ( 1 - r/R)
            self.thickness[:] = h[:] - g.bottomTopographyCell[:]
            self.vorticity[:] = 2*u0/a * xp.sin(g.latCell[:])
            self.divergence[:] = 0.
            
            self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell) / xp.sum(g.areaCell)

            self.curlWind_cell[:] = 0.
            self.divWind_cell[:] = 0.

            
        elif c.test_case == 21:
            # A wind-driven gyre at mid-latitude in the northern hemisphere
            tau0 = 1.e-4
            
            latmin = xp.min(g.latCell[:,0]); latmax = xp.max(g.latCell[:,0])
            lonmin = xp.min(g.lonCell[:,0]); lonmax = xp.max(g.lonCell[:,0])

            latmid = 0.5*(latmin+latmax)
            latwidth = latmax - latmin

            lonmid = 0.5*(lonmin+lonmax)
            lonwidth = lonmax - lonmin

            r = c.sphere_radius

            self.vorticity[:,:] = 0.
            self.divergence[:,:] = 0.

            if c.nLayers == 1:
                self.thickness[:,:] = 4000.
            elif c.nLayers == 2:
                self.thickness[:,0] = 1000.
                self.thickness[:,1] = 3000.
            else:
                raise ValueError('This test only takes nLayers = 1 or 2.')

            self.psi_cell[:,:] = 0.0
            self.psi_vertex[:,:] = 0.0
            
            # Initialize wind
            self.curlWind_cell[:] = -tau0 * np.pi/(latwidth*r) * \
                                    xp.sin(np.pi*(g.latCell[:,0]-latmin) / latwidth)
            self.divWind_cell[:] = 0.


            self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
            topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
            for layer in range(c.nLayers):
                self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg

            print('Sea/layer sufrace average height:')
            print(self.SS0)

            
        elif c.test_case == 22:
            # One gyre with no forcing, for a bounded domain over NA
            d = xp.sqrt(32*(g.latCell[:,:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:,:]-(-1.1))**2/.3**2)
            f0 = xp.mean(g.fCell)

            if c.nLayers == 1:
                self.thickness[:] = 4000.
                self.psi_cell[:] = 2*xp.exp(-d**2) * 0.5*(1-xp.tanh(20*(d-1.5)))
#               self.psi_cell[:] -= np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
                self.psi_cell *= c.gravity / f0 * self.thickness
            elif c.nLayers == 2:
                self.thickness[:,0] = 1000.
                self.thickness[:,1] = 3000.
                self.psi_cell[:,:] = xp.exp(-d**2) * 0.5*(1-xp.tanh(20*(d-1.5)))
#               self.psi_cell[:] -= np.sum(self.psi_cell * g.areaCell) / np.sum(g.areaCell)
                self.psi_cell[:,:] *= c.gravity / f0 * self.thickness[:,:]

            else:
                raise ValueError('This test case only takes nLayers = 1 or 2.')
                
                
            self.phi_cell[:,:] = 0.
            self.vorticity[:,:] = vc.discrete_laplace_v(self.psi_cell[:,:])
            self.vorticity[:,:] /= self.thickness[:,:]
            self.divergence[:,:] = 0.
            
            # Initialize wind
            self.curlWind_cell[:] = 0.
            self.divWind_cell[:] = 0.

            # Eliminate bottom drag
            #c.bottomDrag = 0.

            # Eliminate lateral diffusion
            #c.delVisc = 0.
            #c.del2Visc = 0.
            
            self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
            topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
            for layer in range(c.nLayers):
                self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg

            print('Sea/layer sufrace average height:')
            print(self.SS0)

            
        else:
            raise ValueError("Invaid choice for the test case.")
                                                
        # Set time to zero
        self.time = 0.0
        
    def restart_from_file(self, g, c):
        rdata = nc.Dataset(c.restart_file,'r')

        start_ind = len(rdata.dimensions['Time']) - 1

        self.thickness[:,:] = xp.asarray(rdata.variables['thickness'][start_ind,:,:])
        self.vorticity[:,:] = xp.asarray(rdata.variables['vorticity_cell'][start_ind,:,:])
        self.divergence[:,:] = xp.asarray(rdata.variables['divergence'][start_ind,:,:])
        self.psi_cell[:,:] = xp.asarray(rdata.variables['psi_cell'][start_ind,:,:])
        self.phi_cell[:,:] = xp.asarray(rdata.variables['phi_cell'][start_ind,:,:])
        self.time = rdata.variables['xtime'][start_ind]

        g.bottomTopographyCell[:,0] = xp.asarray(rdata.variables['bottomTopographyCell'][:])
        self.SS0[:] = xp.sum(self.thickness * g.areaCell, axis=0) / xp.sum(g.areaCell, axis=0)
        topo_avg = xp.sum(g.bottomTopographyCell * g.areaCell, axis=0).item()/xp.sum(g.areaCell, axis=0).item()
        for layer in range(c.nLayers):
            self.SS0[layer] = xp.sum(self.SS0[layer:]) + topo_avg
        
#        self.SS0 = xp.sum((self.thickness + g.bottomTopographyCell) * g.areaCell, axis=0) / xp.sum(g.areaCell)
        
        # Read simulation parameters
        c.test_case = int(rdata.test_case)
        c.dt = float(rdata.dt)
        c.delVisc = float(rdata.delVisc)
        c.bottomDrag = float(rdata.bottomDrag)
        c.sphere_radius = float(rdata.sphere_radius)
        
        rdata.close( )

    def initialization(self, poisson, g, vc, c):

        if c.do_restart:
            self.restart_from_file(g,c)
        else:
            self.start_from_function(vc, g, c)


        # Compute diagnostic variables
        self.compute_diagnostics(poisson, g, vc, c)
            
        # Open the output file and create new state variables
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        out.createDimension('nVertLevels', c.nLayers)
        out.createDimension('Time', None)
        out.createVariable('xtime', 'f8', ('Time',))
        out.createVariable('thickness', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('vorticity_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('divergence', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('psi_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('phi_cell', 'f8', ('Time', 'nCells', 'nVertLevels'))
        out.createVariable('nVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('tVelocity', 'f8', ('Time', 'nEdges', 'nVertLevels'))
        out.createVariable('kenergy', 'f8', ('Time', 'nCells', 'nVertLevels'))
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
        out.kappa = "%e" % (c.kappa)
        if c.on_a_sphere:
            out.on_a_sphere = "YES"
        else:
            out.on_a_sphere = "NO"
        out.on_a_global_sphere = "%s" % (c.on_a_global_sphere)
        out.sphere_radius = "%e" % (c.sphere_radius)
        out.no_flux_BC = "%s" % (c.no_flux_BC)
        out.no_slip_BC = "%s" % (c.no_slip_BC)
        out.free_slip_BC = "%s" % (c.free_slip_BC)
        
        out.close( )
        

    def compute_tendencies(self, g, c, vc):

        # Tendency for thicknetss
        self.vCell[:,:] = self.phi_cell[:,:] - 0.5 * c.GM_kappa * self.thickness[:,:]**2
        self.tend_thickness[:] = -vc.discrete_laplace_v(self.vCell)
        self.tend_thickness[:] += c.delVisc * vc.discrete_laplace_v(self.thickness)

        # Tendency for vorticity
        if c.conserve_enstrophy:
            pv_vertex = vc.cell2vertex(self.pv_cell)
            psi_edge = vc.cell2edge(self.psi_cell)

            self.vEdge[:] = vc.discrete_grad_n(self.psi_cell)
            self.vEdge *= self.pv_edge
            self.vEdge -= psi_edge * vc.discrete_grad_n(self.pv_cell)
            self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
            self.tend_vorticity[:] = 1./6 * vc.vertex2cell(self.vVertex)

            self.vEdge[:] = psi_edge * vc.discrete_skewgrad_nd(pv_vertex)  # valid on a globe
            self.vEdge[:] -= self.pv_edge * vc.discrete_skewgrad_nd(self.psi_vertex)
            self.tend_vorticity += 1./6 * vc.discrete_div_v(self.vEdge)

            self.vEdge[:] = vc.discrete_skewgrad_nd(pv_vertex) * vc.discrete_grad_n(self.psi_cell)
            self.vEdge -= vc.discrete_skewgrad_nd(self.psi_vertex) * vc.discrete_grad_n(self.pv_cell)
            self.tend_vorticity += 1./3 * vc.edge2cell(self.vEdge)

        else:
            self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
            self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
            self.tend_vorticity[:] = 0.5 * vc.vertex2cell(self.vVertex)

            self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad_nd(self.psi_vertex)
            self.tend_vorticity[:] -= 0.5 * vc.discrete_div_v(self.vEdge)
                
        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
        self.tend_vorticity[:] -= vc.discrete_div_v(self.vEdge)

        self.tend_vorticity[:,0] += self.curlWind_cell / self.thickness[:,0]
        self.tend_vorticity[:,-1] -= c.bottomDrag * self.vorticity[:,-1]
        self.tend_vorticity[:] += c.delVisc * vc.discrete_laplace_v(self.vorticity)

        # Tendency for divergence
        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.psi_cell)
        self.tend_divergence[:] = vc.discrete_div_v(self.vEdge)

        self.vEdge[:] = self.pv_edge * vc.discrete_grad_n(self.phi_cell)
        self.vVertex[:] = vc.discrete_curl_t(self.vEdge)
        self.tend_divergence[:] += 0.5 * vc.vertex2cell(self.vVertex)

#        self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad_nd(self.phi_vertex)
#        self.vEdge[:] = self.pv_edge * vc.discrete_skewgrad_nn(self.phi_vertex)  # phi satisfies homog. Neumann

        # The following lines implement the natural BC's for skewgrad; natural BC's are needed to
        # strictly retain the symmetry of the Poisson bracket. However, phi_vertex satisfies the homogeneous Neumann
        # BC's. In this case, the requirement for symmetry may be slightly relaxed, and the above skewgrad_nn be used
        # instead, which is simpler.
        self.vEdge[:] = cmp.discrete_skewgrad_nnat(self.phi_vertex, self.phi_cell, g.verticesOnEdge, g.cellsOnEdge, \
                                                   g.dvEdge)
        self.vEdge *= self.pv_edge
        self.tend_divergence[:] -= 0.5 * vc.discrete_div_v(self.vEdge)

        ## The boundary terms
        if not c.on_a_global_sphere:
            pv_bv_edge = 0.5*(self.pv_cell[vc.cellBoundary_ord[:-1]-1,:] + self.pv_cell[vc.cellBoundary_ord[1:]-1,:])
            phi_diff_edge = self.phi_cell[vc.cellBoundary_ord[1:]-1,:] - self.phi_cell[vc.cellBoundary_ord[:-1]-1,:]
            pv_phi_diff_edge = pv_bv_edge * phi_diff_edge
            self.tend_divergence[vc.cellBoundary_ord[0]-1,:] -= 1./4/g.areaCell[vc.cellBoundary_ord[0]-1,:] * \
                    (pv_phi_diff_edge[-1,:] + pv_phi_diff_edge[0,:])
            self.tend_divergence[vc.cellBoundary_ord[1:-1]-1,:] -= 1./4/g.areaCell[vc.cellBoundary_ord[1:-1]-1,:] * \
                    (pv_phi_diff_edge[:-1,:] + pv_phi_diff_edge[1:,:])

        self.tend_divergence[:] -= vc.discrete_laplace_v(self.geoPot)

        self.tend_divergence[:,0] += self.divWind_cell / self.thickness[:,0] 
        self.tend_divergence[:,-1] -= c.bottomDrag * self.divergence[:,-1] 
        self.tend_divergence[:] += c.delVisc * vc.discrete_laplace_v(self.divergence)
        
    def compute_diagnostics(self, poisson, g, vc, c):

        if c.test_case == 1:
            #For shallow water test case #1, reset the vorticity and divergence to the initial states
            a = c.sphere_radius
            u0 = 2*np.pi*a / (12*86400)
            self.vorticity[:] = 2*u0/a * xp.sin(g.latCell[:])
            self.divergence[:] = 0.

        if xp.any(self.thickness[:,0] < 0.):
            raise ValueError('Negative layer thickness detected. Aborting.')
            
        self.thickness_edge[:] = vc.cell2edge(self.thickness)
#        self.compute_psi_phi(vc, g, c)
        for layer in range(c.nLayers):
            self.compute_psi_phi_cpl2(poisson, vc, g, c, layer)

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
            pass
        
        # Compute the absolute vorticity
        self.eta_cell = self.vorticity + g.fCell

        # Compute the potential vorticity
        self.pv_cell = self.eta_cell / self.thickness
        
        # Map from cell to edge
        self.pv_edge[:] = vc.cell2edge(self.pv_cell)

        # Compute kinetic energy on the edge
        self.compute_kenergy_edge(vc, g, c)
        self.kenergy[:] = vc.edge2cell(self.kenergy_edge)

        # Compute the Montgomery potential. Only one of the following should
        # be active.
        ## Completely decoupled non-interactive layers
#        self.geoPot = c.rho_vec * (g.bottomTopographyCell + self.thickness)
#        self.geoPot *= c.gravity / c.rho_vec
#        self.geoPot += self.kenergy

        ## Completely decoupled non-interactive layers; but second layer is forced
#        self.geoPot = c.rho_vec * (g.bottomTopographyCell + self.thickness)
#        self.geoPot[:,1] += c.rho_vec[1] * self.l1Thickness
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy

        ## One-directional interaction; layer 2 forced by layer 1.
#        self.geoPot = c.rho_vec * (g.bottomTopographyCell + self.thickness)
#        self.geoPot[:,1] += c.rho_vec[0] * self.thickness[:,0]
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy

        ## One-directional interaction; layer 1 forced by layer 2
#        self.geoPot = c.rho_vec * (g.bottomTopographyCell + self.thickness)
#        self.geoPot[:,0] += c.rho_vec[0] * self.thickness[:,1]
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy
        
        ## 2-layer with bi-directional interaction
#        if c.nLayers != 2:
#            raise ValueError('Only 2-layer case is considered.')
#        else:
#            self.geoPot = c.rho_vec * (g.bottomTopographyCell + self.thickness)
#            self.geoPot[:,0] += c.rho_vec[0] * self.thickness[:,1]
#            self.geoPot[:,1] += c.rho_vec[0] * self.thickness[:,0]
#            self.geoPot *= c.gravity
#            self.geoPot /= c.rho_vec
#            self.geoPot += self.kenergy
       
        ## Interactive layers (Implementation #1, Boussinesq)
#        self.geoPot = c.rho_vec * g.bottomTopographyCell
#        for i in range(c.nLayers):
#            self.geoPot[:,i] += xp.sum(c.rho_vec[:i] * self.thickness[:,:i], axis = 1)
#            self.geoPot[:,i] += c.rho_vec[i] * xp.sum(self.thickness[:,i:], axis = 1)
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy

        ## Interactive layers (Implementation #2, Non-Boussinesq)
#        self.geoPot = c.rho_vec * g.bottomTopographyCell
#        for i in range(c.nLayers):
#            self.geoPot[:,i] += xp.sum(c.rho_vec[:i] * self.thickness[:,:i], axis = 1)
#            self.geoPot[:,i] += c.rho_vec[i] * xp.sum(self.thickness[:,i:], axis = 1)
#        self.geoPot *= c.gravity / c.rho_vec
#        self.geoPot += self.kenergy
        
        ## Interactive layers (Implementation #3, Boussinesq)
#        self.geoPot[:,0]  = c.rho_vec[0] * (xp.sum(self.thickness, axis=1) + g.bottomTopographyCell[:,0])
#        for k in range(1,c.nLayers):
#            self.geoPot[:,k] = self.geoPot[:,k-1] + (c.rho_vec[k]-c.rho_vec[k-1]) *  \
#                (xp.sum(self.thickness[:,k:], axis = 1) + g.bottomTopographyCell[:,0])
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy


        ## Interactive layers (Implementation #3, Boussinesq, average depth subtracted)
        self.geoPot[:,0]  = c.rho_vec[0] * (xp.sum(self.thickness, axis=1) + g.bottomTopographyCell[:,0] - self.SS0[0])
        for k in range(1,c.nLayers):
            self.geoPot[:,k] = self.geoPot[:,k-1] + (c.rho_vec[k]-c.rho_vec[k-1]) *  \
                (xp.sum(self.thickness[:,k:], axis = 1) + g.bottomTopographyCell[:,0] - self.SS0[k])
        self.geoPot *= c.gravity / c.rho0
        self.geoPot -= c.kappa*vc.discrete_laplace_v(self.geoPot)     # Artifical PE
        self.geoPot += self.kenergy
        ## Potential energy (power function) due to layer thinning; sigma = 2e7
        #self.geoPot -= c.power*c.sigma/c.min_thickness*(c.min_thickness / self.thickness[:,:])**(c.power+1)
        #self.art_energy = c.sigma*xp.sum(xp.sum(c.min_thickness/self.thickness[:,:]*g.areaCell[:,:]))
        ## Potential energy (exponential function) due to layer thinning; sigma = 2e6, min_thickness = 100.
        #self.geoPot -= c.sigma/c.min_thickness * np.exp(-self.thickness[:,:]/c.min_thickness)
        #self.art_energy = c.sigma*xp.sum(xp.sum(xp.exp(-self.thickness[:,:]/c.min_thickness)*g.areaCell[:,:]))
        ## Potential energy (Gaussian) due to layer thinning; sigma = 2e10
        #self.geoPot -= 2*c.sigma/c.min_thickness**2 * self.thickness[:,:]*np.exp(-(self.thickness[:,:]/c.min_thickness)**2)
        #self.art_energy = c.sigma*xp.sum(xp.sum(xp.exp(-(self.thickness[:,:]/c.min_thickness)**2)*g.areaCell[:,:]))
        ## Potential energy (powered Gaussian) due to layer thinning; sigma = 2e112
        #self.geoPot -= 2*c.power*c.sigma*self.thickness[:,:]**(2*c.power-1)/c.min_thickness**(2*c.power) * np.exp(-(self.thickness[:,:]/c.min_thickness)**(2*c.power))
        #self.art_energy = c.sigma*xp.sum(xp.sum(xp.exp(-(self.thickness[:,:]/c.min_thickness)**(2*c.power))*g.areaCell[:,:]))

        ## Interactive layers (Implementation #3, Boussinesq, average depth subtracted)
        ## with artificial potential energy
#        self.geoPot[:,0]  = c.rho_vec[0] * (xp.sum(self.thickness, axis=1) + g.bottomTopographyCell[:,0] \
#                                            - self.SS0[0]) - c.power*c.sigma/c.min_thickness*(c.min_thickness/self.thickness[:,0])**(c.power+1)
#        for k in range(1,c.nLayers):
#            self.geoPot[:,k] = self.geoPot[:,k-1] + (c.rho_vec[k]-c.rho_vec[k-1]) *  \
#                (xp.sum(self.thickness[:,k:], axis = 1) + g.bottomTopographyCell[:,0] - self.SS0[k]) \
#                 - c.power*c.sigma/c.min_thickness*(c.min_thickness/self.thickness[:,k])**(c.power+1)
#        self.geoPot *= c.gravity / c.rho0
#        self.geoPot += self.kenergy
        
        ## Interactive layers (Implementation #4, Non-Boussinesq)
#        self.geoPot[:,0]  = c.rho_vec[0] * (xp.sum(self.thickness, axis=1) + g.bottomTopographyCell[:,0])
#        for k in range(1,c.nLayers):
#            self.geoPot[:,k] = self.geoPot[:,k-1] + (c.rho_vec[k]-c.rho_vec[k-1]) *  \
#                (xp.sum(self.thickness[:,k:], axis = 1) + g.bottomTopographyCell[:,0])
#        self.geoPot *= c.gravity / c.rho_vec
#        self.geoPot += self.kenergy

        # Compute kinetic energy, total energy, and potential enstrophy
        self.kinetic_energy = xp.sum(xp.sum(self.kenergy_edge * self.thickness_edge * g.areaEdge))

        ## DEBUG ##
#        if self.kinetic_energy < 0.:
#            raise ValueError('Negative energy!!')

        # Compute the real and artificial potential energy
        self.vCell[:, 0] = xp.sum(self.thickness[:,0:], axis=1) + g.bottomTopographyCell[:,0] - self.SS0[0]
        self.vEdge[:,0] = vc.discrete_grad_n(self.vCell[:,0])
        self.pot_energy = 0.5 * c.gravity * c.rho_vec[0]/c.rho0 * xp.sum(self.vCell[:,0]**2 * g.areaCell[:,0]).item( )
        self.art_energy = c.gravity * c.rho_vec[0]/c.rho0 * xp.sum(self.vEdge[:,0]**2 * g.areaEdge[:,0]).item( ) 
        for iLayer in range(1,c.nLayers):
            self.vCell[:,0] = xp.sum(self.thickness[:,iLayer:], axis=1) + g.bottomTopographyCell[:,0] - self.SS0[iLayer]
            self.vEdge[:,0] = vc.discrete_grad_n(self.vCell[:,0])
            d_rho = c.rho_vec[iLayer] - c.rho_vec[iLayer-1]
            self.pot_energy += 0.5 * c.gravity * d_rho /c.rho0 * xp.sum(self.vCell[:,0]**2 * g.areaCell[:,0]).item( )
            self.art_energy += c.gravity * d_rho /c.rho0 * xp.sum(self.vEdge[:,0]**2 * g.areaEdge[:,0]).item( )
        self.art_energy *= c.kappa

        # Compute the artificial potential energy
#        self.pot_energy = 0.5 * c.kappa * c.gravity * c.rho_vec[0]/c.rho0 * xp.sum(vc.discrete_grad_n(xp.sum(self.thickness[:,0:], axis=1) + \
#        	g.bottomTopographyCell[:,0] - self.SS0[0])**2 * g.areaCell[:,0], axis=0).item( )
#        for iLayer in range(1,c.nLayers):
#        	self.pot_energy += 0.5 * c.gravity * (c.rho_vec[iLayer] - c.rho_vec[iLayer-1])/c.rho0 * \
#        		xp.sum( (xp.sum(self.thickness[:,iLayer:], axis=1) + \
#        			g.bottomTopographyCell[:,0] - self.SS0[iLayer])**2 * g.areaCell[:,0])
                

        self.pot_enstrophy = 0.5 * xp.sum(xp.sum(g.areaCell[:] * self.thickness * self.pv_cell[:]**2))
        

    def compute_psi_phi(self, vc, g, c):
        # To compute the psi_cell and phi_cell, requires EllipticCpl object

        # Update the coefficient matrix for the coupled system
        vc.update_matrix_for_coupled_elliptic(self.thickness_edge, c, g)

        # Prepare the right-hand side and initial solution
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
            self.vortdiv[g.nCells] = 0.            # Set first element to zeor to make phi_cell[0] zero

        vc.POcpl.solve(vc.coefM, self.vortdiv, self.psiphi, linear_solver = c.linear_solver)
        self.psi_cell[:] = self.psiphi[:g.nCells]
        self.phi_cell[:] = self.psiphi[g.nCells:]


    def compute_psi_phi_cpl2(self, poisson, vc, g, c, layer):
        # To compute psi_cell and phi_cell, requires the EllipticCpl2 object (that is, poisson)
        
        # Update the coefficient matrix for the coupled system
        poisson.update(self.thickness_edge[:,layer], vc, c, g)
        
        # Prepare the right-hand side and initial solution
        self.circulation[:] = self.vorticity[:,layer] * g.areaCell[:,0]
        self.flux[:] = self.divergence[:,layer] * g.areaCell[:,0]
        
        if c.on_a_global_sphere:
            # A global domain with no boundary
            self.circulation[0] = 0.   # Set first element to zeor to make psi_cell[0] zero
            self.flux[0] = 0.   # Set first element to zeor to make phi_cell[0] zero
            
        else:
            # A bounded domain with homogeneous Dirichlet for the psi and
            # homogeneous Neumann for phi
            self.circulation[vc.cellBoundary-1] = 0.   # Set boundary elements to zeor to make psi_cell zero there
            self.flux[0] = 0.                   # Set first element to zeor to make phi_cell[0] zero

        layer_psi_cell = self.psi_cell[:,layer].copy()
        layer_phi_cell = self.phi_cell[:,layer].copy()
        poisson.solve(self.circulation, self.flux, layer_psi_cell, layer_phi_cell)
        self.psi_cell[:,layer] = layer_psi_cell
        self.phi_cell[:,layer] = layer_phi_cell
        
    def save(self, c, g, k):
        # Open the output file to save current data data
        out = nc.Dataset(c.output_file, 'a', format='NETCDF3_64BIT')
        
        out.variables['xtime'][k] = self.time
        if c.use_gpu:
            out.variables['thickness'][k,:,:] = self.thickness.get()
            out.variables['vorticity_cell'][k,:,:] = self.vorticity.get()
            out.variables['divergence'][k,:,:] = self.divergence.get()
            out.variables['psi_cell'][k,:,:] = self.psi_cell.get()
            out.variables['phi_cell'][k,:,:] = self.phi_cell.get()
            out.variables['nVelocity'][k,:,:] = self.nVelocity.get()
            out.variables['tVelocity'][k,:,:] = self.tVelocity.get()
            out.variables['kenergy'][k,:,:] = self.kenergy.get()

        else:    
            out.variables['thickness'][k,:,:] = self.thickness
            out.variables['vorticity_cell'][k,:,:] = self.vorticity
            out.variables['divergence'][k,:,:] = self.divergence
            out.variables['psi_cell'][k,:,:] = self.psi_cell
            out.variables['phi_cell'][k,:,:] = self.phi_cell
            out.variables['nVelocity'][k,:,:] = self.nVelocity
            out.variables['tVelocity'][k,:,:] = self.tVelocity
            out.variables['kenergy'][k,:,:] = self.kenergy

            
        if k==0:
            if c.use_gpu:
                out.variables['curlWind_cell'][:] = self.curlWind_cell.get()
                out.variables['bottomTopographyCell'][:] = g.bottomTopographyCell.get()
            else:
                out.variables['curlWind_cell'][:] = self.curlWind_cell[:]
                out.variables['bottomTopographyCell'][:] = g.bottomTopographyCell[:]
                
        out.close( )

    def compute_tc2_errors(self, iStep, s_init, error1, error2, errorInf, g):
        # For test case #2, compute the errors
        error1[iStep+1, 0, :] = xp.sum(xp.abs(self.thickness[:] - s_init.thickness[:])*g.areaCell[:], axis=0) / xp.sum(xp.abs(s_init.thickness[:])*g.areaCell[:], axis=0)
        error1[iStep+1, 1, :] = xp.sum(xp.abs(self.vorticity[:] - s_init.vorticity[:])*g.areaCell[:], axis=0) / xp.sum(xp.abs(s_init.vorticity[:])*g.areaCell[:], axis=0)
        error1[iStep+1, 2, :] = xp.max(xp.abs(self.divergence[:] - s_init.divergence[:]), axis=0) 

        error2[iStep+1, 0, :] = xp.sqrt(xp.sum((self.thickness[:] - s_init.thickness[:])**2*g.areaCell[:], axis=0))
        error2[iStep+1, 0, :] /= xp.sqrt(xp.sum((s_init.thickness[:])**2*g.areaCell[:], axis=0))
        error2[iStep+1, 1, :] = xp.sqrt(xp.sum((self.vorticity[:] - s_init.vorticity[:])**2*g.areaCell[:], axis=0))
        error2[iStep+1, 1, :] /= xp.sqrt(xp.sum((s_init.vorticity[:])**2*g.areaCell[:], axis=0))
        error2[iStep+1, 2, :] = xp.sqrt(xp.sum((self.divergence[:] - s_init.divergence[:])**2*g.areaCell[:], axis=0))
        error2[iStep+1, 2, :] /= xp.sqrt(xp.sum(g.areaCell[:], axis=0))

        errorInf[iStep+1, 0, :] = xp.max(xp.abs(self.thickness[:] - s_init.thickness[:]), axis=0) / xp.max(xp.abs(s_init.thickness[:]), axis=0)
        errorInf[iStep+1, 1, :] = xp.max(xp.abs(self.vorticity[:] - s_init.vorticity[:]), axis=0) / xp.max(xp.abs(s_init.vorticity[:]), axis=0)
        errorInf[iStep+1, 2, :] = xp.max(xp.abs(self.divergence[:] - s_init.divergence[:]), axis=0)

    def compute_kenergy_edge(self, vc, g, c):
        # Compute the kinetic energy
        self.vEdge = vc.discrete_skewgrad_t(self.psi_cell)
        self.kenergy_edge[:] = self.vEdge**2

        self.vEdge = vc.discrete_grad_n(self.phi_cell)
        self.kenergy_edge += self.vEdge**2

        self.kenergy_edge += vc.discrete_skewgrad_nd(self.psi_vertex) * vc.discrete_grad_n(self.phi_cell)
        self.kenergy_edge += vc.discrete_skewgrad_t(self.psi_cell) * vc.discrete_grad_tn(self.phi_vertex)

        self.kenergy_edge /= self.thickness_edge**2
        
        ## DEBUGGING ##
#        for iLayer in range(c.nLayers):
#            print("min and max of kenergy_edge: %f, %f" % (xp.min(self.kenergy_edge[:,iLayer]), xp.max(self.kenergy_edge[:,iLayer])))
#            if self.kenergy_edge[:,iLayer].min() < 0.:
#                raise ValueError('kenergy_edge negative')
        
def timestepping_rk4_z_hex(s, s_pre, s_old, poisson, g, vc, c):

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

            s_intm.compute_diagnostics(poisson, g, vc, c)

    # Prediction using the latest s_intm values
    s.psi_cell[:] = s_intm.psi_cell[:]
    s.phi_cell[:] = s_intm.phi_cell[:]

    s.compute_diagnostics(poisson, g, vc, c)

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


    s.compute_diagnostics(poisson, g, vc, c)

    
