import numpy as np
#from scipy.sparse import csr_matrix

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
    
