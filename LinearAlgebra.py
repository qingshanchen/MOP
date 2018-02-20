import numpy as np

class MaxItersError(Exception):
    pass

def cg(env, A, b, x, max_iter=1000, relres=1e-5):

    r = b - A.dot(x)
    p = r.copy()
    Ap = np.zeros(np.size(p))
    dx = np.zeros(np.size(p))
    r2 = np.dot(r,r)
    res0 = np.sqrt(np.sum(np.dot(b,b)))
    res = np.sqrt(r2)

    if res0 < np.finfo('float32').tiny:
        # Zero right-hand; the solution should be zero
        x[:] = 0.
        return 0, 0
    elif res/res0 < relres:
        # Initial guess close enough to the true solution
        return 0, 0

    x_norm = np.sqrt(np.dot(x,x)) + np.finfo('float32').tiny

    counter = 0
    while True:
        # Punch the attendance card
        counter += 1

        Ap[:] = A.dot(p[:])
        alpha = r2 / np.dot(p, Ap)
        dx = alpha*p
        x += dx
        r -= alpha*Ap
        r2_new = np.dot(r,r)
        beta = r2_new / r2
        p = r + beta * p
        dx_norm = np.sqrt(np.dot(dx,dx))

        # Check if the target tol has been reached
#        res = np.sqrt(r2_new)
#        if res/res0 < relres:
        if dx_norm / x_norm < relres:
            return 0, counter
        
        if counter > max_iter:
#            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
            raise MaxItersError
        
        r2 = r2_new
        x_norm = np.sqrt(np.dot(x,x))


def cudaCG(env, PO, b, x, max_iter=1000, relres=1e-5):

    # Initial copying data from host to device
    dr = env.cuda.to_device(b)
    dx = env.cuda.to_device(x)

    # Generate two local variables on device
    dp = env.cuda.device_array_like(b)
    env.cuBlas.scal(0., dp)       # Initialize to zero
    dAp = env.cuda.device_array_like(b)
    env.cuBlas.scal(0., dAp)
    x_incr = env.cuda.device_array_like(b)
    env.cuBlas.scal(0., x_incr)


    # Compute the l2 norm of the right-hand side
    res0 = env.cuBlas.nrm2(dr)

    # Compute r = r - A.x
    env.cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = -1., descr=PO.cuSparseDescr, \
                   csrVal=PO.dData, csrRowPtr=PO.dPtr, csrColInd=PO.dInd, x=dx, beta=1., y=dr)

    # Compute p = r
    env.cuBlas.axpy(1., dr, dp)

    # compute l2 norm of the initial residual
    res = env.cuBlas.nrm2(dr)
    r2 = res * res

    if res0 < np.finfo('float32').tiny:
        # Zero right-hand; the solution should be zero
        x[:] = 0.
        return 0, 0
    
    elif res/res0 < relres:
        # Initial guess close enough to the true solution; nothing to do
        return 0, 0

    x_norm = env.cuBlas.nrm2(dx) + np.finfo('float32').tiny

    # reset the counter
    counter = 0
    while True:
        
        # Punch the attendance card
        counter += 1

        ## Group 1 execution:
        # Compute Ap = A.p
        env.cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = 1., descr=PO.cuSparseDescr, \
                       csrVal=PO.dData, csrRowPtr=PO.dPtr, csrColInd=PO.dInd, x=dp, beta=0., y=dAp)

        # Set x_incr = 0.
        env.cuBlas.scal(0., x_incr)

        ## Group 2 execution:
        # Compute alpha
        alpha = r2 / env.cuBlas.dot(dp, dAp)

        ## Group 3 execution:
        # x_incr = alpha * dp
        env.cuBlas.axpy(alpha, dp, x_incr)

        # Compute x = x + alpha * dp
        env.cuBlas.axpy(alpha, dp, dx)

        # Compute r = r - alpha.Ap
        env.cuBlas.axpy(-alpha, dAp, dr)

        ## Group 4
        # Compute l2 norm of the residual
        x_incr_norm = env.cuBlas.nrm2(x_incr)


        # Check if the target tol has been reached
#        if res/res0 < relres:
        if x_incr_norm / x_norm < relres:

            dx.copy_to_host(x)
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))

        ## Group 5
        res = env.cuBlas.nrm2(dr)
        x_norm = env.cuBlas.nrm2(dx)

        ## Group 6
        # Compute beta
        r2_new = res*res
        beta = r2_new / r2

        ## Group 7
        # Compute p = beta*p
        env.cuBlas.scal(beta, dp)
        ## Group 8
        # dp = dp + dr
        env.cuBlas.axpy(1., dr, dp)
        
        r2 = r2_new


def cudaPCG(env, PO, b, x, max_iter=1000, relres=1e-5):

    # Initial copying data from host to device 
    db = env.cuda.device_array_like(b)
    dr = env.cuda.device_array_like(b)
    dx = env.cuda.device_array_like(x)

    ### Initial transformations of b and x
    # Set dx = LT.x
    env.cuSparse.csrmv(trans='N', m=PO.LT.shape[0], n=PO.LT.shape[1], nnz=PO.LT.nnz,
                   alpha = 1., descr=PO.LTmv_descr, csrVal=PO.LTdata, \
                   csrRowPtr=PO.LTptr, csrColInd=PO.LTind, \
                   x=x, beta=0., y=dx)

    # Set db = L\b
    env.cuSparse.csrsv_solve(trans='N', m=PO.L.shape[0], alpha=1.0, \
                         descr=PO.Lsv_descr, csrVal=PO.Ldata, \
                         csrRowPtr=PO.Lptr, csrColInd=PO.Lind, info=PO.Lsv_info, x=b, y=db)
    

    # Generate two local variables on device
    dp = env.cuda.device_array_like(b)
    env.cuBlas.scal(0., dp)       # Initialize to zero
    dAp = env.cuda.device_array_like(b)
    env.cuBlas.scal(0., dAp)


    # Compute the l2 norm of the right-hand side
    res0 = env.cuBlas.nrm2(db)

    # Compute dr = db - L\ A. LT\ dx
    env.cuSparse.csrsv_solve(trans='T', m=PO.L.shape[0], alpha=1.0, \
                         descr=PO.Lsv_descr, csrVal=PO.Ldata, \
                         csrRowPtr=PO.Lptr, csrColInd=PO.Lind, info=PO.LTsv_info, x=dx, y=dp)  # dp temporarily used here
#    raise ValueError

    env.cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = 1., \
                   descr=PO.Adescr, csrVal=PO.Adata, csrRowPtr=PO.Aptr, csrColInd=PO.Aind, x=dp, \
                   beta=0., y=dAp)   # dAp temporarily used here
#    info = PO.Lsv_info.copy( )
    env.cuSparse.csrsv_solve(trans='N', m=PO.L.shape[0], alpha=1.0, \
                         descr=PO.Lsv_descr, csrVal=PO.Ldata, \
                         csrRowPtr=PO.Lptr, csrColInd=PO.Lind, info=PO.Lsv_info, x=dAp, y=dr)
    env.cuBlas.scal(-1, dr)
    env.cuBlas.axpy(1., db, dr)

    # Set p = r
    env.cuBlas.scal(0., dp)
    env.cuBlas.axpy(1., dr, dp)

    # compute l2 norm of the initial residual
    res = env.cuBlas.nrm2(dr)
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

        # Compute Ap = L\ A. LT\ dp
#        info = PO.LTsv_info.copy( )
#        env.cuBlas.scal(0., dAp)
#        env.cuSparse.csrsv_solve(trans='N', m=PO.LT.shape[0], alpha=1.0, \
#                             descr=PO.LTsv_descr, csrVal=PO.LTdata, \
#                             csrRowPtr=PO.LTptr, csrColInd=PO.LTind, info=PO.LTsv_info, x=dp, y=dAp)   
        env.cuSparse.csrsv_solve(trans='T', m=PO.L.shape[0], alpha=1.0, \
                             descr=PO.Lsv_descr, csrVal=PO.Ldata, \
                             csrRowPtr=PO.Lptr, csrColInd=PO.Lind, info=PO.LTsv_info, x=dp, y=dAp)   
#        raise ValueError
        env.cuSparse.csrmv(trans='N', m=PO.A.shape[0], n=PO.A.shape[1], nnz=PO.A.nnz, alpha = 1., \
                       descr=PO.Adescr, csrVal=PO.Adata, csrRowPtr=PO.Aptr, csrColInd=PO.Aind, \
                       x=dAp, beta=0., y=db)     # db used as a temp variable
#        info = PO.Lsv_info.copy( )
        env.cuSparse.csrsv_solve(trans='N', m=PO.L.shape[0], alpha=1.0, \
                             descr=PO.Lsv_descr, csrVal=PO.Ldata, csrRowPtr=PO.Lptr, csrColInd=PO.Lind, \
                             info=PO.Lsv_info, x=db, y=dAp)


        # Compute alpha
        alpha = r2 / env.cuBlas.dot(dp, dAp)

        # Compute dx = dx + alpha.dp
        env.cuBlas.axpy(alpha, dp, dx)

        # Compute dr = dr - alpha.dAp
        env.cuBlas.axpy(-alpha, dAp, dr)

        # Compute l2 norm of the residual
        res = env.cuBlas.nrm2(dr)
        r2_new = res*res

        # Compute beta
        beta = r2_new / r2

        # Compute dp = beta*dp + r
        env.cuBlas.scal(beta, dp)
        env.cuBlas.axpy(1., dr, dp)

        if res != res:
            raise ValueError("Exceptions detected in cudaPCG")

        # Check if the target tol has been reached
        if res/res0 < relres:
#            info = PO.LTsv_info.copy( )
            env.cuSparse.csrsv_solve(trans='T', m=PO.L.shape[0], alpha=1.0, \
                                 descr=PO.Lsv_descr, csrVal=PO.Ldata, \
                                 csrRowPtr=PO.Lptr, csrColInd=PO.Lind, info=PO.LTsv_info, x=dx, y=x)
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
        
        r2 = r2_new
        
    

def pcg(env, A, L, L_solve, LT, LT_solve, b, x, max_iter=1000, relres=1e-5):

    X = LT.dot(x)    # Use a new variable; x will be used to take the solution back
    B = L_solve(b)   # Use a new variable; b is unchanged.
    
    #r = b - A.dot(x)
    #r = spsolve_triangular(LT, x, lower=False)
    r = LT_solve(X)
    r = A.dot(r)
    #r = spsolve_triangular(L, r)
    r = L_solve(r)
    r = B - r

    p = r.copy()
    Ap = np.zeros(np.size(p))
    r2 = np.dot(r,r)
    res0 = np.sqrt(np.sum(np.dot(B,B)))
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
        X += alpha*p
        r -= alpha*Ap
        r2_new = np.dot(r,r)
        beta = r2_new / r2
        p = r + beta * p

        # Check if the target tol has been reached
        res = np.sqrt(np.sum(r2_new))

        if res/res0 < relres:
            #x = spsolve_triangular(LT, x, lower=False)
            x[:] = LT_solve(X)
            return 0, counter
        
        if counter > max_iter:
            raise ValueError("Maximum number of iteration exceeded. Number of iters: %d. relres = %e" % (counter, res/res0))
        
        r2 = r2_new
    
        
