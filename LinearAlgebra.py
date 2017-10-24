import numpy as np
#from scipy.sparse import csr_matrix

def cg(A, b, x0, max_iter=1000, relres=1e-5):

    counter = 0
    r = b - A.dot(x0)
    p = r.copy()
    Ap = A.dot(p)
    r2 = np.dot(r,r)
    res0 = np.sqrt(np.sum(np.dot(b,b)))
    
    while True:
        # Punch the attendance card
        counter += 1
        
        alpha = r2 / np.dot(p, A.dot(p))
        x0 += alpha*p
        r -= alpha*A.dot(p)
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
    
