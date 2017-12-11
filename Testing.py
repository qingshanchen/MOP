import numpy as np
import time
from LinearAlgebra import cg
from pyamg import rootnode_solver
from pyamg.util.linalg import norm
from numpy import ones, array, arange, zeros, abs, random
from scipy.sparse import isspmatrix_bsr, isspmatrix_csr
from scipy.sparse.linalg import spsolve_triangular, factorized, splu
#from solver_diagnostics import solver_diagnostics
from accelerate import cuda
from copy import deepcopy as deepcopy
import numba

def run_tests(g, vc, c, s):

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
        # To run solver_diagnostics for the AMG
        print("To run solver_diagnostics for the AMG")
        
        A = -1 * vc.D2s
        A = A.tocsr( )

        solver_diagnostics(A, fname='D2s', 
                       cycle_list=['V'],
                       symmetry='symmetric', 
                       definiteness='positive',
                       solver=rootnode_solver)

        A = -1 * vc.E2s 
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
        A = -1 * vc.D2s
        A = A.tocsr( )
        b = A.dot(sol)

#        x1 = np.zeros(g.nCells)
#        t0 = time.clock( )
#        x1[:] = vc.lu_D2s.solve(b)
#        t1 = time.clock( )
#        print("rel error = %f" % (np.sqrt(np.sum((x1+sol)**2))/np.sqrt(np.sum(sol*sol))))
#        print("CPU time for the direct method: %f" % (t1-t0,))

        x4 = np.zeros(g.nCells)
        t0 = time.clock( )
        info, nIter = cg(A, b, x4, relres=c.err_tol)
        t1 = time.clock( )
        print(("info = %d" % info))
        print(("nIter = %d" % nIter))
        print(("rel error = %f" % (np.sqrt(np.sum((x4-sol)**2))/np.sqrt(np.sum(sol*sol)))))
        print(("CPU time for cg solver: %f" % (t1-t0,)))

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
        print(("rel error = %f" % (np.sqrt(np.sum((x-sol)**2))/np.sqrt(np.sum(sol*sol)))))
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


    if True:
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
