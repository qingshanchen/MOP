import numpy as np
from scipy.sparse.linalg import spsolve_triangular
from accelerate.cuda.sparse import Sparse as cuSparseClass
cuSparse = cuSparseClass( )
from accelerate.cuda.blas import Blas as cuBlasClass
cuBlas = cuBlasClass( )
from numba import cuda
import time


def cg(A, b, x0, max_iter=1000, relres=1e-5):

    r = b - A.dot(x0)
    p = r.copy()
    Ap = np.zeros(np.size(p))
    r2 = np.dot(r,r)
    res0 = np.sqrt(np.sum(np.dot(b,b)))
    res = np.sqrt(np.sum(r2))

    if res0 < np.finfo('float32').tiny:
        # Zero right-hand; the solution should be zero
        x0[:] = 0.
        return 0, 0
    elif res/res0 < relres:
        # Initial guess close enough to the true solution
        return 0, 0

    counter = 0
    while True:
        # Punch the attendance card
        counter += 1

        Ap[:] = A.dot(p[:])
        alpha = r2 / np.dot(p, Ap)
        x0 += alpha*p
        r -= alpha*Ap
        r2_new = np.dot(r,r)
        beta = r2_new / r2
        p = r + beta * p

        # Check if the target tol has been reached
        res = np.sqrt(np.sum(r2_new))
        if res/res0 < relres:
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
        
        r2 = r2_new


def cudaCG(PO, b, x, max_iter=1000, relres=1e-5):

    # Initial copying data from host to device
    dr = cuda.to_device(b)
    dx = cuda.to_device(x)

    # Generate two local variables on device
    dp = cuda.device_array_like(b)
    cuBlas.scal(0., dp)       # Initialize to zero
    dAp = cuda.device_array_like(b)
    cuBlas.scal(0., dAp)


    # Compute the l2 norm of the right-hand side
    res0 = cuBlas.nrm2(dr)

    # Compute r = r - A.x
    cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = -1., descr=PO.cuSparseDescr, \
                   csrVal=PO.dData, csrRowPtr=PO.dPtr, csrColInd=PO.dInd, x=dx, beta=1., y=dr)

    # Compute p = r
    cuBlas.axpy(1., dr, dp)

    # compute l2 norm of the initial residual
    res = cuBlas.nrm2(dr)
    r2 = res * res

    if res0 < np.finfo('float32').tiny:
        # Zero right-hand; the solution should be zero
        x[:] = 0.
        return 0, 0
    
    elif res/res0 < relres:
        # Initial guess close enough to the true solution; nothing to do
        return 0, 0

    # reset the counter
    counter = 0
    while True:
        
        # Punch the attendance card
        counter += 1

        # Compute Ap = A.p
        cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = 1., descr=PO.cuSparseDescr, \
                       csrVal=PO.dData, csrRowPtr=PO.dPtr, csrColInd=PO.dInd, x=dp, beta=0., y=dAp)

        # Compute alpha
        alpha = r2 / cuBlas.dot(dp, dAp)

        # Compute x = x + alpha.p
        cuBlas.axpy(alpha, dp, dx)

        # Compute r = r - alpha.Ap
        cuBlas.axpy(-alpha, dAp, dr)

        # Compute l2 norm of the residual
        res = cuBlas.nrm2(dr)
        r2_new = res*res

        # Compute beta
        beta = r2_new / r2

        # Compute p = beta*p + r
        cuBlas.scal(beta, dp)
        cuBlas.axpy(1., dr, dp)

        # Check if the target tol has been reached
        if res/res0 < relres:

            dx.copy_to_host(x)
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
        
        r2 = r2_new

    

def pcg(A, L, L_solve, LT, LT_solve, b, x, max_iter=1000, relres=1e-5):

    x = LT.dot(x)
#    b = spsolve_triangular(L, b)
    b = L_solve(b)
    
    #r = b - A.dot(x)
    #r = spsolve_triangular(LT, x, lower=False)
    r = LT_solve(x)
    r = A.dot(r)
    #r = spsolve_triangular(L, r)
    r = L_solve(r)
    r = b - r

    p = r.copy()
    Ap = np.zeros(np.size(p))
    r2 = np.dot(r,r)
    res0 = np.sqrt(np.sum(np.dot(b,b)))
    res = np.sqrt(np.sum(r2))

    if res0 < np.finfo('float32').tiny:
        # Zero right-hand; the solution should be zero
        x[:] = 0.
        return 0, 0
    elif res/res0 < relres:
        # Initial guess close enough to the true solution
        return 0, 0

    counter = 0
    while True:
        # Punch the attendance card
        counter += 1

#        Ap[:] = A.dot(p[:])
        #Ap = spsolve_triangular(LT, p, lower=False)
        Ap = LT_solve(p)
        Ap = A.dot(Ap)
        #Ap = spsolve_triangular(L, Ap)
        Ap = L_solve(Ap)
        
        alpha = r2 / np.dot(p, Ap)
        x += alpha*p
        r -= alpha*Ap
        r2_new = np.dot(r,r)
        beta = r2_new / r2
        p = r + beta * p

        # Check if the target tol has been reached
        res = np.sqrt(np.sum(r2_new))
        if res/res0 < relres:
            #x = spsolve_triangular(LT, x, lower=False)
            x = LT_solve(x)
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
        
        r2 = r2_new
    
        
