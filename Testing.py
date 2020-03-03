import numpy as np
import time
from scipy.sparse import isspmatrix_bsr, isspmatrix_csr
from scipy.sparse.linalg import factorized, splu
#from accelerate import cuda
from copy import deepcopy as deepcopy
import numba
from swe_comp import swe_comp as cmp

def run_tests(env, g, vc, c, s):

    if False:   # Test the linear solver the Lapace equation on the interior cells with homogeneous Dirichlet BC's
        psi_cell_true = np.random.rand(vc.nCells)
        psi_cell_true[vc.cellBoundary[:]-1] = 0.0

        vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, g.areaCell, psi_cell_true)

        #compte psi_cell using vc.A and linear solver
        x = vc.lu_D1.solve(vorticity_cell[vc.cellInterior[:]-1])
        psi_cell = np.zeros(g.nCells)
        psi_cell[vc.cellInterior[:]-1] = x[:]

        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell[:])**2 * vc.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * vc.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors for linear solver")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        
        
    if False:
        # Test the linear solver the Lapace equation on the whole domain
        # The solution is set to zero at cell 0.
        # Also test the linear solver for the Poisson equaiton  on a bounded domain with
        # homogeneous Neumann BC's

        psi_cell_true = np.random.rand(g.nCells)
        psi_cell_true[0] = 0.
        
        vorticity_cell = cmp.discrete_laplace_cell(g.cellsOnEdge, \
            g.dcEdge, g.dvEdge, vc.areaCell, psi_cell_true)

        # Artificially set vorticity_cell[0] to 0
        vorticity_cell[0] = 0.

        #compte psi_cell using vc.A and linear solver
        psi_cell = vc.lu_D2.solve(vorticity_cell[:])

        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors for linear solver")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # using compute_psi_phi
        # The solution is set to zero at cell 0.

        s.thickness[:] = 4000. + np.random.rand(g.nCells) * 100
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.update_coefficient_matrix(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for updating matrix: %f" % (cpu1-cpu0,)))
        print(("Wall time for updating matrix: %f" % (wall1-wall0,)))

        psi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        if c.on_a_global_sphere:
            psi_cell_true[0] = 0.
        else:
            psi_cell_true[vc.cellBoundary[:]-1] = 0.
            
        phi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        phi_cell_true[0] = 0.

        psi_vertex = vc.cell2vertex(psi_cell_true)

        hu = vc.discrete_grad_n(phi_cell_true)
        hu -= vc.discrete_grad_td(psi_vertex)
        u = hu / s.thickness_edge
        s.vorticity = vc.vertex2cell(vc.discrete_curl_trig(u))
        s.divergence = vc.discrete_div(u)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

        l8 = np.max(np.abs(phi_cell_true[:] - s.phi_cell[:])) / np.max(np.abs(phi_cell_true[:]))
        l2 = np.sum(np.abs(phi_cell_true[:] - s.phi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(phi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in phi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # using solvers directly.
        # The solution is set to zero at cell 0.
        from scipy.sparse.linalg import spsolve
        import pyamgx

        s.thickness[:] = 4000. + np.random.rand(g.nCells) * 100
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        vc.update_matrix_for_coupled_elliptic(s.thickness_edge, c, g)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for updating matrix: %f" % (cpu1-cpu0,)))
        print(("Wall time for updating matrix: %f" % (wall1-wall0,)))

        psi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        if c.on_a_global_sphere:
            psi_cell_true[0] = 0.
        else:
            psi_cell_true[vc.cellBoundary[:]-1] = 0.
            
        phi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        phi_cell_true[0] = 0.

        psi_vertex = vc.cell2vertex(psi_cell_true)

        hu = vc.discrete_grad_n(phi_cell_true)
        hu -= vc.discrete_grad_td(psi_vertex)
        u = hu / s.thickness_edge
        s.vorticity = vc.vertex2cell(vc.discrete_curl_trig(u))
        s.divergence = vc.discrete_div(u)

        # To prepare the rhs
        s.vortdiv[:g.nCells] = s.vorticity * g.areaCell
        s.vortdiv[g.nCells:] = s.divergence * g.areaCell
        if c.on_a_global_sphere:
            # A global domain with no boundary
            s.vortdiv[0] = 0.   # Set first element to zeor to make psi_cell[0] zero
        else:
            # A bounded domain with homogeneous Dirichlet for the psi and
            # homogeneous Neumann for phi
            s.vortdiv[vc.cellBoundary-1] = 0.   # Set boundary elements to zeor to make psi_cell zero there
        s.vortdiv[g.nCells] = 0.   # Set first element to zeor to make phi_cell[0] zero
        
        ## Solve the linear system by the direct method
        #cpu0 = time.clock( )
        #wall0 = time.time( )
        #x = spsolve(vc.coefM, s.vortdiv)
        #cpu1 = time.clock( )
        #wall1 = time.time( )

        ## Solve the linear system by AMGX
        import pyamgx
        pyamgx.initialize( )
        # Initialize config, resources and mode:
        #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json')
        cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json') 
        rsc = pyamgx.Resources().create_simple(cfg)
        mode = 'dDDI'

        # Create matrices and vectors:
        d_A = pyamgx.Matrix().create(rsc, mode)
        d_x = pyamgx.Vector().create(rsc, mode)
        d_b = pyamgx.Vector().create(rsc, mode)

        # Create solver:
        slv = pyamgx.Solver().create(rsc, cfg, mode)

        
        #vc.update_matrix_for_coupled_elliptic(
        d_A.upload_CSR(vc.coefM)
        d_A.replace_coefficients(vc.coefM.data)

        d_b.upload(s.vortdiv)
        x = np.zeros(g.nCells*2)
        d_x.upload(x)

        # Setup and solve system:
        cpu0 = time.clock( )
        wall0 = time.time( )
        slv.setup(d_A)
        slv.solve(d_b, d_x)
        cpu1 = time.clock( )
        wall1 = time.time( )

        d_x.download(x)

        # Clean up:
        d_A.destroy()
        d_x.destroy()
        d_b.destroy()
        slv.destroy()
        rsc.destroy()
        cfg.destroy()
        pyamgx.finalize()
        
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        s.psi_cell[:] = x[:g.nCells]
        s.phi_cell[:] = x[g.nCells:]
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

        l8 = np.max(np.abs(phi_cell_true[:] - s.phi_cell[:])) / np.max(np.abs(phi_cell_true[:]))
        l2 = np.sum(np.abs(phi_cell_true[:] - s.phi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(phi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in phi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        
        

    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # The Hamiltonian is defined using the tangential component
        # using the EllipticCPL object
        # vorticity and divergence derived from psi and phi

        s.thickness[:] = 4000. + np.random.rand(g.nCells) * 100
        s.thickness_edge = vc.cell2edge(s.thickness)

        psi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        if c.on_a_global_sphere:
            psi_cell_true[0] = 0.
        else:
            psi_cell_true[vc.cellBoundary[:]-1] = 0.
            
        phi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        phi_cell_true[0] = 0.

        phi_vertex = vc.cell2vertex(phi_cell_true)

        hv = vc.discrete_grad_n(psi_cell_true)
        hv += vc.discrete_grad_tn(phi_vertex)
        v = hv / s.thickness_edge
        s.vorticity = vc.discrete_curl_v(v)
        s.divergence = vc.vertex2cell(vc.discrete_div_t(v))

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

        l8 = np.max(np.abs(phi_cell_true[:] - s.phi_cell[:])) / np.max(np.abs(phi_cell_true[:]))
        l2 = np.sum(np.abs(phi_cell_true[:] - s.phi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(phi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in phi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # The Hamiltonian is defined using the tangential component
        # using the EllipticCPL object
        # vorticity and divergence taken from SWSTC #2

        a = c.sphere_radius
        u0 = 2*np.pi*a / (12*86400)
        gh0 = 2.94e4
        gh = np.sin(g.latCell[:])**2
        gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
        s.thickness[:] = gh / c.gravity
        h0 = gh0 / c.gravity

        s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
        s.divergence[:] = 0.

        psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
        psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
        psi_cell_true -= psi_cell_true[0]
        phi_cell_true = np.zeros(g.nCells)
        
        s.thickness_edge = vc.cell2edge(s.thickness)
        
        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = %e" % l8)
        print("L^2 error        = %e" % l2)        
        

    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # Both normal and tangential components are used to define the Hamiltonian
        # using the EllipticCPL object
        # vorticity and divergence derived from psi and phi

        s.thickness[:] = 4000. + np.random.rand(g.nCells) * 100
        s.thickness_edge = vc.cell2edge(s.thickness)

        psi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        if c.on_a_global_sphere:
            psi_cell_true[0] = 0.
        else:
            psi_cell_true[vc.cellBoundary[:]-1] = 0.
            
        phi_cell_true = np.random.rand(g.nCells) * 2.4e+9
        phi_cell_true[0] = 0.

        psi_vertex = vc.cell2vertex(psi_cell_true)
        phi_vertex = vc.cell2vertex(phi_cell_true)

        hv = vc.discrete_grad_n(psi_cell_true)
        hv += vc.discrete_grad_tn(phi_vertex)
        v = hv / s.thickness_edge

        hu = vc.discrete_grad_n(phi_cell_true)
        hu -= vc.discrete_grad_td(psi_vertex)
        u = hu / s.thickness_edge
        
        s.vorticity = 0.5*vc.discrete_curl_v(v)
        s.vorticity += 0.5 * vc.vertex2cell(vc.discrete_curl_t(u))
        
        s.divergence = 0.5*vc.vertex2cell(vc.discrete_div_t(v))
        s.divergence += 0.5*vc.discrete_div_v(u)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

        l8 = np.max(np.abs(phi_cell_true[:] - s.phi_cell[:])) / np.max(np.abs(phi_cell_true[:]))
        l2 = np.sum(np.abs(phi_cell_true[:] - s.phi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(phi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in phi")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on the whole domain
        # using the EllipticCPL object
        # vorticity and divergence come from SWSTC #2

        a = c.sphere_radius
        u0 = 2*np.pi*a / (12*86400)
        gh0 = 2.94e4
        gh = np.sin(g.latCell[:])**2
        gh = -(a*c.Omega0*u0 + 0.5*u0*u0)*gh + gh0
        s.thickness[:] = gh / c.gravity
        h0 = gh0 / c.gravity

        s.vorticity[:] = 2*u0/a * np.sin(g.latCell[:])
        s.divergence[:] = 0.

        psi_cell_true = -a * h0 * u0 * np.sin(g.latCell[:]) 
        psi_cell_true[:] += a*u0/c.gravity * (a*c.Omega0*u0 + 0.5*u0**2) * (np.sin(g.latCell[:]))**3 / 3.
        psi_cell_true -= psi_cell_true[0]
        phi_cell_true = np.zeros(g.nCells)
        
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = %e" % l8)
        print("L^2 error        = %e" % l2)        


    if False:
        # Test the linear solver for the coupled elliptic equation on a bounded domain
        # using the EllipticCPL object
        # vorticity and divergence come from TC #12

        latmin = np.min(g.latCell[:]); latmax = np.max(g.latCell[:])
        lonmin = np.min(g.lonCell[:]); lonmax = np.max(g.lonCell[:])

        latmid = 0.5*(latmin+latmax)
        latwidth = latmax - latmin

        lonmid = 0.5*(lonmin+lonmax)
        lonwidth = lonmax - lonmin

        pi = np.pi; sin = np.sin; exp = np.exp
        r = c.sphere_radius
        
        d = np.sqrt(32*(g.latCell[:] - latmid)**2/latwidth**2 + 4*(g.lonCell[:]-(-1.1))**2/.3**2)
        f0 = np.mean(g.fCell)
        s.thickness[:] = 4000.
        psi_cell_true = 2*np.exp(-d**2) * 0.5*(1-np.tanh(20*(d-1.5)))
        psi_cell_true *= c.gravity / f0 * s.thickness
        phi_cell_true = np.zeros(g.nCells)
        s.vorticity = vc.discrete_laplace_v(psi_cell_true)
        s.vorticity /= s.thickness
        s.divergence[:] = 0.

        s.psi_cell[:] = 0.
        s.phi_cell[:] = 0.
        s.thickness_edge = vc.cell2edge(s.thickness)

        cpu0 = time.clock( )
        wall0 = time.time( )
        s.compute_psi_phi(vc, g, c)
        cpu1 = time.clock( )
        wall1 = time.time( )
        print(("CPU time for solving system: %f" % (cpu1-cpu0,)))
        print(("Wall time for solving system: %f" % (wall1-wall0,)))
        
        # Compute the errors
        l8 = np.max(np.abs(psi_cell_true[:] - s.psi_cell[:])) / np.max(np.abs(psi_cell_true[:]))
        l2 = np.sum(np.abs(psi_cell_true[:] - s.psi_cell[:])**2 * g.areaCell[:])
        l2 /=  np.sum(np.abs(psi_cell_true[:])**2 * g.areaCell[:])
        l2 = np.sqrt(l2)
        print("Errors in psi")
        print("L infinity error = %e" % l8)
        print("L^2 error        = %e" % l2)        
        
        
    if False:   # Test the linear solver for the Poisson equation on the triangles with homogeneous Dirichlet BC's
        psi_vertex_true = np.random.rand(g.nVertices)

        vorticity_vertex = cmp.discrete_laplace_vertex(g.verticesOnEdge,  \
                         g.dcEdge, g.dvEdge, g.areaTriangle, psi_vertex_true, 0)

        #compte psi_vertex using linear solver
        psi_vertex = vc.lu_E1.solve(vorticity_vertex)

        # Compute the errors
        l8 = np.max(np.abs(psi_vertex_true[:] - psi_vertex[:])) / np.max(np.abs(psi_vertex_true[:]))
        l2 = np.sum(np.abs(psi_vertex_true[:] - psi_vertex[:])**2 * g.areaTriangle[:])
        l2 /=  np.sum(np.abs(psi_vertex_true[:])**2 * g.areaTriangle[:])
        l2 = np.sqrt(l2)
        print("Errors for the solver for the Poisson with Neumann BC's")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

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
        psi_vertex = vc.lu_E2.solve(vorticity_vertex)

        # Compute the errors
        l8 = np.max(np.abs(psi_vertex_true[:] - psi_vertex[:])) / np.max(np.abs(psi_vertex_true[:]))
        l2 = np.sum(np.abs(psi_vertex_true[:] - psi_vertex[:])**2 * g.areaTriangle[:])
        l2 /=  np.sum(np.abs(psi_vertex_true[:])**2 * g.areaTriangle[:])
        l2 = np.sqrt(l2)
        print("Errors for the solver for the Poisson with Neumann BC's")
        print("L infinity error = ", l8)
        print("L^2 error        = ", l2)        

    if False:
        # To test and compare direct and iterative linear solvers for systems on the primary mesh
        print("To test and compare direct and iterative linear solvers for systems on the primary mesh")
        
        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = vc.D2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        x1[:] = vc.lu_D2s.solve(b)
        t1 = time.clock( )
        print(("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        x2, info = sp.cg(vc.D2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for scipy cg solver: %f" % (t1-t0,)))


        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        A = vc.D2s.tocsr( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))

    if False:
        # To test and compare cg and cudaCG for systems on the primary mesh
        print("To test and compare cg and cudaCG for systems on the primary mesh")
        
        sol = np.random.rand(g.nCells)
#        sol = np.ones(g.nCells)
        sol[0] = 0.
        b = vc.D2s.dot(sol)


        x1 = np.zeros(g.nCells)
        t0 = time.clock( )
        x1[:] = vc.lu_D2s.solve(b)
        t1 = time.clock( )
        print(("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))

        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        A = vc.D2s.tocsr( )
        info, nIter = cg(env, A, b, x4, relres=c.err_tol, max_iter=c.max_iter)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))


#        t0 = time.clock( )
#        x4 = np.zeros(g.nCells)
#        A = vc.D2s.tocsr( )
#        A = -A
#        b = -b
#        info, nIter = pcg(A, vc.D2sL, vc.D2sL_solve, vc.D2sLT, vc.D2sLT_solve, b, x4, max_iter=c.max_iter, relres = c.err_tol)
#        t1 = time.clock( )
#        print(("info = %d" % info))
#        print(("nIter = %d" % nIter))
#        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
#        print(("CPU time for pcg solver: %f" % (t1-t0,)))

        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        info, nIter = cudaCG(env, vc.POpn, b, x2, relres=c.err_tol, max_iter=c.max_iter)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cudaCG solver: %f" % (t1-t0,)))


        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        b = -b
        info, nIter = cudaPCG(env, vc.POpnSPD, b, x1, relres=c.err_tol, max_iter=c.max_iter)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cudaPCG solver: %f" % (t1-t0,)))

        raise ValueError
        
    if False:
        # To run solver_diagnostics for the AMG
        print("To run solver_diagnostics for the AMG")
        
        solver_diagnostics(vc.POpn.A_spd, fname='p15km', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)

        solver_diagnostics(vc.POdn.A_spd, fname='d15km', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)
        
    if False:
        # Timing tests for AMG solvers
        print("Timing tests for AMG solvers ")

        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = vc.POpn.A_spd.dot(sol)

#        x4 = np.zeros(g.nCells)
#        t0 = time.clock( )
#        info, nIter = cg(vc.POpn.A_spd, b, x4, relres=c.err_tol)
#        t1 = time.clock( )
#        print(("info = %d" % info))
#        print(("nIter = %d" % nIter))
#        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
#        print(("CPU time for cg solver: %f" % (t1-t0,)))

        res = []
        x0 = np.zeros(g.nCells)
        t0 = time.clock()
        x = vc.POpn.A_amg.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel="cg", maxiter=300, cycle="V")
        t1 = time.clock()
        print(("rel error = %e" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print("nIter = %d" % len(res))
        print(("CPU time for AMG cg solver: %f" % (t1-t0,)))

        
    if False:
        print("To test and compare direct and iterative linear solvers for systems on the primary mesh")
        
        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = vc.D2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        x1[:] = vc.lu_D2s.solve(b)
        t1 = time.clock( )
        print(("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        x2, info = sp.cg(vc.D2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for scipy cg solver: %f" % (t1-t0,)))


        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        A = vc.D2s.tocsr( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))
        

    if False:
        # To test and compare direct and iterative linear solvers for systems on the dual mesh
        print("To test and compare direct and iterative linear solvers for systems on the dual mesh")
        
        sol = np.random.rand(g.nVertices)
        sol[0] = 0.
        b = vc.E2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nVertices)
        x1[:] = vc.lu_E2s.solve(b)
        t1 = time.clock( )
        print(("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nVertices)
        x2, info = sp.cg(vc.E2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for scipy cg solver: %f" % (t1-t0,)))


        A = vc.E2s.tocsr( )
        t0 = time.clock( )
        x4 = np.zeros(g.nVertices)
        info, nIter = cg(vc.E2s, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))


    if False:
        print("To study and compare initializaiton schemes ")
        
        sol_cell = np.cos(g.latCell)*np.sin(g.lonCell)
        sol_cell[:] -= sol_cell[0]
        sol_vertex = np.cos(g.latVertex)*np.sin(g.lonVertex)
        sol_vertex[:] -= sol_vertex[0]

        vort_cell = cmp.discrete_laplace( \
                 g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, \
                                          sol_cell)
        vort_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, vort_cell)
        
        x_cell = np.zeros(g.nCells)
        b_cell = vort_cell[:] * g.areaCell[:]
        b_cell[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.D2s, b_cell, x_cell, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell)))))
        print(("CPU time for cg solver on primary mesh: %f" % (t1-t0,)))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = vort_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,)))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print(("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,)))
        

    if False:
        print("Repeat previous test, but using data from SWSTC #5 ")

        s.initialization(g, c)
        sol_cell = s.psi_cell[:]
        sol_cell[:] -= sol_cell[0]
        sol_vertex = s.psi_vertex[:]
        sol_vertex[:] -= sol_vertex[0]

        x_cell = np.zeros(g.nCells)
        b_cell = s.vorticity[:] * g.areaCell[:]
        b_cell[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.D2s, b_cell, x_cell, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell)))))
        print(("CPU time for cg solver on primary mesh: %f" % (t1-t0,)))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = s.vorticity_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,)))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print(("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        t0 = time.clock( )
        info, nIter = cg(vc.E2s, b_vertex, x_vertex, max_iter=c.max_iter, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex)))))
        print(("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,)))
        
        
    if False:
        # Test LinearAlgebra.cg solver by a small simple example
        A = csr_matrix([[2,1,0],[1,3,1],[0,1,4]])
        sol = np.array([1,0,-1])
        b = np.array([2,0,-4])

        x0 = np.array([0.,0.,0.])
        x, info = cg_scipy(A, b, x0=x0, maxiter = 200)
        print(("cg_scipy: info = %d" % info))
        print(x)
        info, nIter = cg(A, b, x0, max_iter = 100)
        print(("nIter = %d" % nIter))
        print("x = ")
        print(x0)


    if False:
        print("To test cg with incomplete cholesky as preconditioner")

        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = vc.D2s.dot(sol)
        
        A = vc.D2s.tocsr( )
        A.eliminate_zeros( )
        A_t = deepcopy(A)
        A_t.data = np.where(A_t.nonzero()[0] >= A_t.nonzero()[1], A_t.data, 0.)
        A_t.eliminate_zeros( )
        R = A_t.copy( )
        R = -R

        
        cuSparse = cuda.sparse.Sparse()
        D2s_descr = cuSparse.matdescr(matrixtype='S', fillmode='L')
        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
                                       descr=D2s_descr, csrVal=R.data, \
                                       csrRowPtr=R.indptr, csrColInd=R.indices)
        cuSparse.csric0(trans='N', m=R.shape[0], \
                        descr=D2s_descr, csrValM=R.data, csrRowPtrA=R.indptr,\
                        csrColIndA=R.indices, info=info)
#        cuSparse.csrilu0(trans='N', m=R.shape[0], \
#                        descr=D2s_descr, csrValM=R.data, csrRowPtrA=R.indptr,\
#                        csrColIndA=R.indices, info=info)
        
        # Test triangular solver 
        b1 = R.dot(sol)
        Rsolve = factorized(R)
#        lu_R = splu(R, permc_spec='NATURAL')
        t0 = time.clock( )
        #x = spsolve_triangular(R, b1, lower=True, overwrite_b=False, overwrite_A=False)
        x = Rsolve(b1)
        #x = lu_R.solve(b1)
        t1 = time.clock( )
        print(("rel error for triangular solver = %e" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for a direct solver for triangular solver: %f" % (t1-t0,)))


        Rdata = numba.cuda.to_device(R.data)
        Rptr = numba.cuda.to_device(R.indptr)
        Rind = numba.cuda.to_device(R.indices)
        R_descr = cuSparse.matdescr(matrixtype='T', fillmode='L')
#        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
#                                       descr=R_descr, csrVal=R.data, \
#                                       csrRowPtr=R.indptr, csrColInd=R.indices)
        info = cuSparse.csrsv_analysis(trans='N', m=R.shape[0], nnz=R.nnz, \
                                       descr=R_descr, csrVal=Rdata, \
                                       csrRowPtr=Rptr, csrColInd=Rind)        
        b1 = R.dot(sol)
        x = np.zeros(np.size(b1))
        t0 = time.clock( )
#        cuSparse.csrsv_solve(trans='N', m=R.shape[0], alpha=1.0, \
#                             descr=R_descr, csrVal=R.data, \
#                             csrRowPtr=R.indptr, csrColInd=R.indices, info=info, x=b1, y=x)
        cuSparse.csrsv_solve(trans='N', m=R.shape[0], alpha=1.0, \
                             descr=R_descr, csrVal=Rdata, \
                             csrRowPtr=Rptr, csrColInd=Rind, info=info, x=b1, y=x)
        t1 = time.clock( )
        print(("rel error for triangular solver = %e" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for a cuda solver for triangular solver: %f" % (t1-t0,)))

        
        D2s_solve = factorized(vc.D2s)
        x1 = np.zeros(g.nCells)
        t0 = time.clock( )
        #x1[:] = vc.lu_D2s.solve(b)
        x1[:] = D2s_solve(b)
        t1 = time.clock( )
        print(("rel error = %e" % (np.sqrt(np.sum((x1-sol)**2)))))
        print(("CPU time for the direct method: %f" % (t1-t0,)))
        
        x4 = np.zeros(g.nCells)
        t0 = time.clock( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %e" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))

        
        raise ValueError("Stop for checking.")

    elif False:
        # To test the AMGX solver for the Poisson equation on primal mesh
        
        import pyamgx
        import os

        pyamgx.initialize()

        # Initialize config, resources and mode:
        #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json')   # Best for 2621442 prim
        cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json') 

        rsc = pyamgx.Resources().create_simple(cfg)
        mode = 'dDDI'

        # Create matrices and vectors:
        A = pyamgx.Matrix().create(rsc, mode)
        x = pyamgx.Vector().create(rsc, mode)
        b = pyamgx.Vector().create(rsc, mode)

        # Create solver:
        slv = pyamgx.Solver().create(rsc, cfg, mode)

        hA = vc.POpn.A
        A.upload(hA.indptr, hA.indices, hA.data)
        slv.setup(A)

        for k in range(5):
            sol = np.random.rand(hA.shape[0])
            sol[0] = 0.
            h_b = hA.dot(sol)
            h_x = np.zeros(np.size(h_b))

            b.upload( h_b)
            x.upload( h_x)

            # Setup and solve system:
            slv.solve(b, x)

            x.download(h_x)
            print(("rel error for pyamg solver = %e" % (np.sqrt(np.sum((h_x-sol)**2))/np.sqrt(np.sum(sol*sol)))))

        # Clean up:
        A.destroy()
        x.destroy()
        b.destroy()
        slv.destroy()
        rsc.destroy()
        cfg.destroy()

        pyamgx.finalize()


    elif False:
        # To test the AMGX solver for the coupled Poisson equation
        
        import pyamgx
        import os

        pyamgx.initialize()

        # Initialize config, resources and mode:
        #cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json')   # Best for 2621442 prim
        cfg = pyamgx.Config().create_from_file('amgx_config/PCGF_AGGREGATION_JACOBI.json') 

        rsc = pyamgx.Resources().create_simple(cfg)
        mode = 'dDDI'

        # Create matrices and vectors:
        A = pyamgx.Matrix().create(rsc, mode)
        x = pyamgx.Vector().create(rsc, mode)
        b = pyamgx.Vector().create(rsc, mode)

        # Create solver:
        slv = pyamgx.Solver().create(rsc, cfg, mode)

        
        #vc.update_matrix_for_coupled_elliptic(
        hA = vc.POpn.A
        A.upload(hA.indptr, hA.indices, hA.data)
        slv.setup(A)

        for k in range(5):
            sol = np.random.rand(hA.shape[0])
            sol[0] = 0.
            h_b = hA.dot(sol)
            h_x = np.zeros(np.size(h_b))

            b.upload( h_b)
            x.upload( h_x)

            # Setup and solve system:
            slv.solve(b, x)

            x.download(h_x)
            print(("rel error for pyamg solver = %e" % (np.sqrt(np.sum((h_x-sol)**2))/np.sqrt(np.sum(sol*sol)))))

        # Clean up:
        A.destroy()
        x.destroy()
        b.destroy()
        slv.destroy()
        rsc.destroy()
        cfg.destroy()

        pyamgx.finalize()

        
    elif False:
        # To compare the performances of AMGX and pyAMG
        
        import pyamgx
        import os

        pyamgx.initialize()

        # Initialize config, resources and mode:
        cfg = pyamgx.Config().create_from_file('PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json')   # Best for 2621442 prim
        rsc = pyamgx.Resources().create_simple(cfg)
        mode = 'dDDI'

        # Create matrices and vectors:
        A = pyamgx.Matrix().create(rsc, mode)
        x = pyamgx.Vector().create(rsc, mode)
        b = pyamgx.Vector().create(rsc, mode)

        # Create solver:
        slv = pyamgx.Solver().create(rsc, cfg, mode)

        hA = vc.POpn.A
        # Read system from file
        A.upload(hA.shape[0], hA.nnz, hA.indptr, hA.indices, hA.data)
        slv.setup(A)

        for k in range(5):
            sol = np.random.rand(hA.shape[0])
            sol[0] = 0.
            h_b = hA.dot(sol)
            h_x = np.zeros(np.size(h_b))

            res = []
            b1 = -h_b.copy( )
            x0 = np.zeros(hA.shape[0])
            t0 = time.time()
            sol1 = vc.POpn.A_amg.solve(b1, x0=x0, tol=1e-6, residuals=res, accel="cg", maxiter=300, cycle="V")
            t1 = time.time()
            print(("rel error for pyamg = %e" % (np.sqrt(np.sum((sol1-sol)**2))/np.sqrt(np.sum(sol*sol)))))
            print("nIter = %d" % len(res))
            print(("Wall time for AMG cg solver: %f" % (t1-t0,)))


            t0 = time.time( )
            b.upload(hA.shape[0], h_b)
            x.upload(hA.shape[0], h_x)

            # Setup and solve system:
            slv.solve(b, x)
            x.download(h_x)
            t1 = time.time( )
            print(("rel error for amgx solver = %e" % (np.sqrt(np.sum((h_x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
            print(("Wall time for AMGX solver: %f" % (t1-t0,)))

        # Clean up:
        A.destroy()
        x.destroy()
        b.destroy()
        slv.destroy()
        rsc.destroy()
        cfg.destroy()

        pyamgx.finalize()

        
    elif False:
        # Compare scipy dot with cuda mv.
        
        from scipy.sparse import tril

        x = np.random.rand(g.nCells)

        cuSparse = cuda.sparse.Sparse()
        #D2s = tril(vc.D2s, format='csr')
        D2s = vc.POpn.A
        data = numba.cuda.to_device(D2s.data)
        ptr = numba.cuda.to_device(D2s.indptr)
        ind = numba.cuda.to_device(D2s.indices)
        #D2s_descr = cuSparse.matdescr(matrixtype='S', fillmode='L')
        D2s_descr = cuSparse.matdescr( )

        t0a = time.clock( )
        t0b = time.time( )
        y = D2s.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for dot: %f" % (t1a-t0a,)))
        print(("Wall time for dot: %f" % (t1b-t0b,)))

        # Create arrays on host, and transfer them to device
        y1 = np.zeros(np.size(x))

        t0a = time.clock( )
        t0b = time.time()
        xd = numba.cuda.to_device(x)
        d_y1 = numba.cuda.to_device(y1)
        t1b = time.time()
        print(("Wall time for transfering vector to device: %f" % (t1b-t0b,)))
        
        t1a = time.clock( )
        t1b = time.time()
        cuSparse.csrmv(trans='N', m=D2s.shape[0], n=D2s.shape[1], nnz=D2s.nnz, alpha=1.0, \
                             descr=D2s_descr, csrVal=data, \
                             csrRowPtr=ptr, csrColInd=ind, x=xd, beta=0., y=d_y1)
        t2a = time.clock( )
        t2b = time.time( )
        print(("CPU time for cuda-mv: %f" % (t2a-t1a,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))

        d_y1.copy_to_host(y1)
        t3a = time.clock( )
        t3b = time.time( )
        print(("Wall time for transfering back to host: %f" % (t3b-t2b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y)**2))/np.sqrt(np.sum(y*y)))))
        print(("Total wall time for cuda-mv: %f" % (t3b-t0b,)))

        

    elif False:
        # Compare discrete_div and mDiv (as matrix-vector product), and GPU mv with d_mDiv
        cuSparse = cuda.sparse.Sparse()
        
        x = np.random.rand(g.nEdges)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for discrete_div: %f" % (t1a-t0a,)))
        print(("Wall time for discrete_div: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.mDiv.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for mDiv: %f" % (t1a-t0a,)))
        print(("Wall time for mDiv: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

        # Create arrays on host, and transfer them to device
        y1[:] = 0.

        t0a = time.clock( )
        t0b = time.time( )
        d_x = numba.cuda.to_device(x)
        d_y1 = numba.cuda.to_device(y1)

        t1a = time.clock( )
        t1b = time.time( )
        cuSparse.csrmv(trans='N', m=vc.d_mDiv.shape[0], n=vc.d_mDiv.shape[1], nnz=vc.d_mDiv.nnz, alpha=1.0, \
                             descr=vc.d_mDiv.cuSparseDescr, csrVal=vc.d_mDiv.dData, \
                             csrRowPtr=vc.d_mDiv.dPtr, csrColInd=vc.d_mDiv.dInd, x=d_x, beta=0., y=d_y1)

        t2a = time.clock( )
        t2b = time.time( )
        d_y1.copy_to_host(y1)


        t3a = time.clock( )
        t3b = time.time( )
        y1 = vc.discrete_div(x)
        t4a = time.clock( )
        t4b = time.time( )
        print(("Wall time for copy to device: %f" % (t1b-t0b,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))
        print(("Wall time for copy to host: %f" % (t3b-t2b,)))
        print(("Wall time for discrete_div: %f" % (t4b-t3b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))


    elif False:
        # Compare discrete_curl and  mCurl (as matrix-vector product), and GPU mv with d_mCurl
        cuSparse = cuda.sparse.Sparse()
        
        x = np.random.rand(g.nEdges)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for discrete_curl: %f" % (t1a-t0a,)))
        print(("Wall time for discrete_curl: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.mCurl.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for mCurl: %f" % (t1a-t0a,)))
        print(("Wall time for mCurl: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

        # Create arrays on host, and transfer them to device
        y1[:] = 0.

        t0a = time.clock( )
        t0b = time.time( )
        d_x = numba.cuda.to_device(x)
        d_y1 = numba.cuda.to_device(y1)

        t1a = time.clock( )
        t1b = time.time( )
        cuSparse.csrmv(trans='N', m=vc.d_mCurl.shape[0], n=vc.d_mCurl.shape[1], nnz=vc.d_mCurl.nnz, alpha=1.0, \
                             descr=vc.d_mCurl.cuSparseDescr, csrVal=vc.d_mCurl.dData, \
                             csrRowPtr=vc.d_mCurl.dPtr, csrColInd=vc.d_mCurl.dInd, x=d_x, beta=0., y=d_y1)

        t2a = time.clock( )
        t2b = time.time( )
        d_y1.copy_to_host(y1)


        t3a = time.clock( )
        t3b = time.time( )
        y1 = vc.discrete_curl(x)
        t4a = time.clock( )
        t4b = time.time( )
        print(("Wall time for copy to device: %f" % (t1b-t0b,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))
        print(("Wall time for copy to host: %f" % (t3b-t2b,)))
        print(("Wall time for discrete_curl: %f" % (t4b-t3b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))


    elif False:
        # Compare discrete_laplace and  mLaplace (as matrix-vector product), and GPU mv with d_mLaplace
        cuSparse = cuda.sparse.Sparse()
        
        x = np.random.rand(g.nCells)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for discrete_laplace: %f" % (t1a-t0a,)))
        print(("Wall time for discrete_laplace: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.mLaplace.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for mLaplace: %f" % (t1a-t0a,)))
        print(("Wall time for mLaplace: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

        # Create arrays on host, and transfer them to device
        y1[:] = 0.

        t0a = time.clock( )
        t0b = time.time( )
        d_x = numba.cuda.to_device(x)
        d_y1 = numba.cuda.to_device(y1)

        t1a = time.clock( )
        t1b = time.time( )
        cuSparse.csrmv(trans='N', m=vc.d_mLaplace.shape[0], n=vc.d_mLaplace.shape[1], nnz=vc.d_mLaplace.nnz, alpha=1.0, \
                             descr=vc.d_mLaplace.cuSparseDescr, csrVal=vc.d_mLaplace.dData, \
                             csrRowPtr=vc.d_mLaplace.dPtr, csrColInd=vc.d_mLaplace.dInd, x=d_x, beta=0., y=d_y1)

        t2a = time.clock( )
        t2b = time.time( )
        d_y1.copy_to_host(y1)
        t3a = time.clock( )
        t3b = time.time( )
        print(("Wall time for copy to device: %f" % (t1b-t0b,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))
        print(("Wall time for copy to host: %f" % (t3b-t2b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))


        y2 = vc.discrete_laplace(x)
        t4a = time.clock( )
        t4b = time.time( )
        print(("Wall time for vc.discrete_laplace: %f" % (t4b-t3b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y2-y0)**2))/np.sqrt(np.sum(y0*y0)))))
        

    elif True:
        print("Compare scipy sparse dot and cupy sparse dot, using the mLaplace sparse matrix")
        
        x = np.random.rand(g.nCells)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = vc.mLaplace_v.dot(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for mLaplace: %f" % (t1a-t0a,)))
        print(("Wall time for mLaplace: %f" % (t1b-t0b,)))

        # Create arrays on host, and transfer them to device
        y1 = np.zeros(g.nCells)

        import cupy as cp
        import cupyx
        d_mLaplace = cupyx.scipy.sparse.csr_matrix(vc.mLaplace_v)
        
        t0a = time.clock( )
        t0b = time.time( )
        d_x = cp.array(x)
        d_y1 = cp.array(y1)
        t1a = time.clock( )
        t1b = time.time( )
        
        d_y1 = d_mLaplace.dot(d_x)
        t2a = time.clock( )
        t2b = time.time( )
        
        y1 = d_y1.get( )
        t3a = time.clock( )
        t3b = time.time( )
        
        print(("Wall time for copy to device: %f" % (t1b-t0b,)))
        print(("Wall time for cuda-mv: %f" % (t2b-t1b,)))
        print(("Wall time for copy to host: %f" % (t3b-t2b,)))
        print(("Wall time for GPU computation: %f" % (t3b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))


    elif False:
        # Compare cell2edge and edge2cell with their Fortran versions
        
        x = np.random.rand(g.nCells)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.cell2edge(g.cellsOnEdge, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for Fortran cell2edge: %f" % (t1a-t0a,)))
        print(("Wall time for Fortran cell2edge: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.cell2edge(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for matrix cell2edge: %f" % (t1a-t0a,)))
        print(("Wall time for matrix cell2edge: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))

        x = np.random.rand(g.nEdges)

        t0a = time.clock( )
        t0b = time.time( )
        y0 = cmp.edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell, x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for Fortran edge2cell: %f" % (t1a-t0a,)))
        print(("Wall time for Fortran edge2cell: %f" % (t1b-t0b,)))

        t0a = time.clock( )
        t0b = time.time( )
        y1 = vc.edge2cell(x)
        t1a = time.clock( )
        t1b = time.time( )
        print(("CPU time for matrix edge2cell: %f" % (t1a-t0a,)))
        print(("Wall time for matrix edge2cell: %f" % (t1b-t0b,)))
        print(("rel error = %e" % (np.sqrt(np.sum((y1-y0)**2))/np.sqrt(np.sum(y0*y0)))))
        
