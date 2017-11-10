import numpy as np
import time
from LinearAlgebra import cg
from pyamg import rootnode_solver
from pyamg.util.linalg import norm
from numpy import ones, array, arange, zeros, abs, random
#from scipy import rand, ravel, log10, kron, eye
#from scipy.io import loadmat
from scipy.sparse import isspmatrix_bsr, isspmatrix_csr
from solver_diagnostics import solver_diagnostics

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

    if False:
        # To test and compare direct and iterative linear solvers for systems on the primary mesh
        print("To test and compare direct and iterative linear solvers for systems on the primary mesh")
        
        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = g.D2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        x1[:] = g.lu_D2s.solve(b)
        t1 = time.clock( )
        print("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2))))
        print("CPU time for the direct method: %f" % (t1-t0,))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        x2, info = sp.cg(g.D2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for scipy cg solver: %f" % (t1-t0,))


        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        A = g.D2s.tocsr( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for cg solver: %f" % (t1-t0,))

    if True:
        # To run solver_diagnostics for the AMG
        print("To run solver_diagnostics for the AMG")
        
        A = -1 * g.D2s
        A = A.tocsr( )

        solver_diagnostics(A, fname='D2s', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)

        A = -1 * g.E2s 
        A = A.tocsr( )

        solver_diagnostics(A, fname='E2s', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)
        
    if False:
        # Timing tests for AMG solvers
        print("Timing tests for AMG solvers ")

#        import D2s40962
        
        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        A = -1 * g.D2s
        A = A.tocsr( )
        b = A.dot(sol)

        x1 = np.zeros(g.nCells)
        t0 = time.clock( )
        x1[:] = g.lu_D2s.solve(b)
        t1 = time.clock( )
        print("rel error = %f" % (np.sqrt(np.sum((x1+sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for the direct method: %f" % (t1-t0,))

        x4 = np.zeros(g.nCells)
        t0 = time.clock( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for cg solver: %f" % (t1-t0,))

        # Generate B
        B = ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()
        ml = rootnode_solver(A, B=B, BH=BH,
            strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
            smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
            improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            aggregate="standard",
            presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            max_levels=15,
            max_coarse=300,
            coarse_solver="pinv")

        res = []
        x0 = np.zeros(g.nCells)
        t0 = time.clock()
        x = ml.solve(b, x0=x0, tol=c.err_tol, residuals=res, accel="cg", maxiter=300, cycle="V")
        t1 = time.clock()
        print("rel error = %f" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for AMG cg solver: %f" % (t1-t0,))

        
    if False:
        # To test and compare direct, cg, and cg + amg linear solvers for systems on the primary mesh
        print("To test and compare direct and iterative linear solvers for systems on the primary mesh")
        
        sol = np.random.rand(g.nCells)
        sol[0] = 0.
        b = g.D2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nCells)
        x1[:] = g.lu_D2s.solve(b)
        t1 = time.clock( )
        print("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2))))
        print("CPU time for the direct method: %f" % (t1-t0,))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nCells)
        x2, info = sp.cg(g.D2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for scipy cg solver: %f" % (t1-t0,))


        t0 = time.clock( )
        x4 = np.zeros(g.nCells)
        A = g.D2s.tocsr( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for cg solver: %f" % (t1-t0,))
        

    if False:
        # To test and compare direct and iterative linear solvers for systems on the dual mesh
        print("To test and compare direct and iterative linear solvers for systems on the dual mesh")
        
        sol = np.random.rand(g.nVertices)
        sol[0] = 0.
        b = g.E2s.dot(sol)

        t0 = time.clock( )
        x1 = np.zeros(g.nVertices)
        x1[:] = g.lu_E2s.solve(b)
        t1 = time.clock( )
        print("rel error = %f" % (np.sqrt(np.sum((x1-sol)**2))))
        print("CPU time for the direct method: %f" % (t1-t0,))
        
        t0 = time.clock( )
        x2 = np.zeros(g.nVertices)
        x2, info = sp.cg(g.E2s, b, x2, tol=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("rel error = %f" % (np.sqrt(np.sum((x2-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for scipy cg solver: %f" % (t1-t0,))


        A = g.E2s.tocsr( )
        t0 = time.clock( )
        x4 = np.zeros(g.nVertices)
        info, nIter = cg(g.E2s, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol))))
        print("CPU time for cg solver: %f" % (t1-t0,))


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
        info, nIter = cg(g.D2s, b_cell, x_cell, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell))))
        print("CPU time for cg solver on primary mesh: %f" % (t1-t0,))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = vort_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(g.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        print("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        t0 = time.clock( )
        info, nIter = cg(g.E2s, b_vertex, x_vertex, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        print("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,))
        

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
        info, nIter = cg(g.D2s, b_cell, x_cell, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_cell-sol_cell)**2))/np.sqrt(np.sum(sol_cell*sol_cell))))
        print("CPU time for cg solver on primary mesh: %f" % (t1-t0,))

        x_vertex = np.zeros(g.nVertices)
        b_vertex = s.vorticity_vertex[:] * g.areaTriangle[:]
        b_vertex[0] = 0.
        t0 = time.clock( )
        info, nIter = cg(g.E2s, b_vertex, x_vertex, max_iter=2000, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        print("CPU time for cg solver on dual mesh with generic initialization: %f" % (t1-t0,))

        x_vertex = cmp.cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.verticesOnEdge, sol_cell)
        x_vertex[:] -= x_vertex[0]
        print("Initial guess, rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        t0 = time.clock( )
        info, nIter = cg(g.E2s, b_vertex, x_vertex, max_iter=c.max_iter, relres=c.err_tol)
        t1 = time.clock( )
        print("info = %d" % info)
        print("nIter = %d" % nIter)
        print("rel error = %f" % (np.sqrt(np.sum((x_vertex-sol_vertex)**2))/np.sqrt(np.sum(sol_vertex*sol_vertex))))
        print("CPU time for cg solver on dual mesh with proper initialization: %f" % (t1-t0,))
        
        
    if False:
        # Test LinearAlgebra.cg solver by a small simple example
        A = csr_matrix([[2,1,0],[1,3,1],[0,1,4]])
        sol = np.array([1,0,-1])
        b = np.array([2,0,-4])

        x0 = np.array([0.,0.,0.])
        x, info = cg_scipy(A, b, x0=x0, maxiter = 200)
        print("cg_scipy: info = %d" % info)
        print(x)
        info, nIter = cg(A, b, x0, max_iter = 100)
        print("nIter = %d" % nIter)
        print("x = ")
        print(x0)

