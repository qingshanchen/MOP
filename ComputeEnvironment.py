class ComputeEnvironment:
    def __init__(self, c):
        if c.linear_solver in ['cg', 'cudaCG', 'cudaPCG']:
            from accelerate.cuda.sparse import Sparse as cuSparseClass
            self.cuSparse = cuSparseClass( )
            from accelerate.cuda.blas import Blas as cuBlasClass
            self.cuBlas = cuBlasClass( )
            from numba import cuda
            self.cuda = cuda

        
