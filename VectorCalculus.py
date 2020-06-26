import numpy as np
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from swe_comp import swe_comp as cmp

class Device_CSR:
    def __init__(self, A, env):
        self.dData = env.cuda.to_device(A.data)
        self.dPtr = env.cuda.to_device(A.indptr)
        self.dInd = env.cuda.to_device(A.indices)
        self.shape = A.shape
        self.nnz = A.nnz
        self.cuSparseDescr = env.cuSparse.matdescr( )
#        self.d_vectOut = env.cuda.device_array


class VectorCalculus:
    def __init__(self, g, c, env):
        self.env = env

        self.linear_solver = c.linear_solver

        self.max_iters = c.max_iters
        self.err_tol = c.err_tol

        self.areaCell = g.areaCell.copy()
        self.areaTriangle = g.areaTriangle.copy()

        #
        # Mesh element indices; These should be in the grid object(?)
        #
        if not c.on_a_global_sphere:
            # Collect non-boundary (interior) cells and put into a vector,
            # and boundary cells into a separate vector
            nCellsBoundary = np.sum(g.boundaryCellMark[:]>0)
            nCellsInterior = g.nCells - nCellsBoundary
            
            self.cellInterior, self.cellBoundary, self.cellRankInterior, \
                cellInner_tmp, cellOuter_tmp, self.cellRankInner, \
                nCellsInner, nCellsOuter = \
                cmp.separate_boundary_interior_inner_cells(nCellsInterior,  \
                nCellsBoundary, c.max_int, g.boundaryCellMark, g.cellsOnCell, g.nEdgesOnCell)
            self.cellInner = cellInner_tmp[:nCellsInner]
            self.cellOuter = cellOuter_tmp[:nCellsOuter]

            self.cellBoundary_ord = cmp.boundary_cells_ordered(\
                                nCellsBoundary, g.boundaryCellMark, g.cellsOnCell)

        else:
            self.cellBoundary = np.array([], dtype='int')

        #
        # Divergence on primal
        #
        # Construct the matrix representing the discrete div on the primal mesh (Voronoi cells)
        # No-flux BCs assumed on the boundary
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mDiv_v = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_v = Device_CSR(self.mDiv_v, env)

        #
        # Divergence on dual (triangle)
        #
        ## Construct the matrix representing the discrete div on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mDiv_t = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_t = Device_CSR(self.mDiv_t, env)

        #
        # Curl on primal
        #
        ## Construct the matrix representing the discrete curl on the primal mesh (Voronoi cells)
        ## No-slip BCs assumed on the boundary.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mCurl_v = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_v = Device_CSR(self.mCurl_v, env)

        #
        # Curl on dual
        #
        ## Construct the matrix representing the discrete curl on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mCurl_t = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_t = Device_CSR(self.mCurl_t, env)

        #
        # Laplace on primal (voronoi)
        #
        ## Construct the matrix representing the discrete Laplace operator the primal
        ## mesh (Voronoi mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, \
                                                  g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nCells))
        self.mLaplace_v = A.tocsr( )

        if c.use_gpu:
            self.d_mLaplace_v = Device_CSR(self.mLaplace_v, env)

        #
        # Laplace on dual (triangle)
        #
        ## Construct the matrix representing the discrete Laplace operator the dual
        ## mesh (triangular mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, \
                                       g.verticesOnEdge, g.dvEdge, g.dcEdge, g.areaTriangle)
                                                  
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nVertices))
        self.mLaplace_t = A.tocsr( )

        if c.use_gpu:
            self.d_mLaplace_t = Device_CSR(self.mLaplace_t, env)

        #
        # Gradient normal
        #
        ## Construct the matrix representing the discrete grad operator along the normal direction.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mGrad_n = A.tocsr( )

        A_n = A.tolil()   
        A_n[:,0] = 0.
        self.mGrad_n_n = A_n.tocsr( )
        self.mGrad_n_n.eliminate_zeros()

        if c.use_gpu:
            self.d_mGrad_n = Device_CSR(self.mGrad_n, env)

        #
        # Gradient tangential(?) with Dirichlet
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Dirichlet BC's 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_td(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_td = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_td = Device_CSR(self.mGrad_td, env)

        #
        # Gradient tangential(?) with Neumann
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Neumann BC's
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_tn(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_tn = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_tn = Device_CSR(self.mGrad_tn, env)

        #
        # Skew gradient tangential
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction.  mSkewgrad_t = mGrad_n
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_t(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_t = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_t = Device_CSR(self.mSkewgrad_t, env)

        #
        # Skew gradient tangential w. Dirichlet
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction. Homogeneous Dirichlet assumed
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_td(g.cellsOnEdge, g.dcEdge, g.boundaryCellMark)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_td = A.tocsr( )
        self.mSkewgrad_td.eliminate_zeros( )
        
        if c.use_gpu:
            self.d_mSkewgrad_td = Device_CSR(self.mSkewgrad_td, env)

        #
        # Skew gradient normal w. Dirichlet
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the normal direction. Homogeneous Dirichlet assumed.
        ## mSkewgrad_n = - mGrad_td
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_nd(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mSkewgrad_nd = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_nd = Device_CSR(self.mSkewgrad_nd, env)

        #
        # Map from cell to vertex
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex = Device_CSR(self.mCell2vertex, env)

        A_n = A.tolil( )
        A_n[:,0] = 0.       # zero for entry 0; Neumann
        self.mCell2vertex_n = A_n.tocsr()
        self.mCell2vertex_n.eliminate_zeros( )

        #
        # Map cell to vertex w. Dirichlet
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh; homogeneous Dirichlet BC's are assumed
        ## On a global sphere, cell 0 is considered the single boundary pt.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex_psi(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, \
                                                 g.boundaryCellMark, c.on_a_global_sphere)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex_psi = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex_psi = Device_CSR(self.mCell2vertex_psi, env)

        #
        # Map vertex to cell
        #
        ## Construct the matrix representing the mapping from the dual mesh onto the primal
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_vertex2cell(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nVertices))
        self.mVertex2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mVertex2cell = Device_CSR(self.mVertex2cell, env)

        #
        # Map cell to edge
        #
        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2edge = Device_CSR(self.mCell2edge, env)

        #
        # Map edge to cell
        # 
        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mEdge2cell = Device_CSR(self.mEdge2cell, env)

        
        ## Some temporary variables as place holders
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)

            
    def discrete_div_v(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mDiv_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mDiv_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mDiv_v.shape[0], \
                n=self.d_mDiv_v.shape[1], nnz=self.d_mDiv_v.nnz, alpha=1.0, \
                descr=self.d_mDiv_v.cuSparseDescr, csrVal=self.d_mDiv_v.dData, \
                csrRowPtr=self.d_mDiv_v.dPtr, csrColInd=self.d_mDiv_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mDiv_v.dot(vEdge)


    def discrete_div_t(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mDiv_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mDiv_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mDiv_t.shape[0], \
                n=self.d_mDiv_t.shape[1], nnz=self.d_mDiv_t.nnz, alpha=1.0, \
                descr=self.d_mDiv_t.cuSparseDescr, csrVal=self.d_mDiv_t.dData, \
                csrRowPtr=self.d_mDiv_t.dPtr, csrColInd=self.d_mDiv_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mDiv_t.dot(vEdge)
        

    def discrete_curl_v(self, vEdge):
        '''
        The discrete curl operator on the primal mesh.
        No-slip boundary conditions implied on the boundary.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl_v.shape[0], \
                n=self.d_mCurl_v.shape[1], nnz=self.d_mCurl_v.nnz, alpha=1.0, \
                descr=self.d_mCurl_v.cuSparseDescr, csrVal=self.d_mCurl_v.dData, \
                csrRowPtr=self.d_mCurl_v.dPtr, csrColInd=self.d_mCurl_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl_v.dot(vEdge)


    def discrete_curl_t(self, vEdge):
        '''
        The discrete curl operator on the dual mesh.
        '''

        if c.use_gpu:
            assert len(vEdge) == self.d_mCurl_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(vEdge)

            sCell = np.zeros(self.d_mCurl_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(sCell)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCurl_t.shape[0], \
                n=self.d_mCurl_t.shape[1], nnz=self.d_mCurl_t.nnz, alpha=1.0, \
                descr=self.d_mCurl_t.cuSparseDescr, csrVal=self.d_mCurl_t.dData, \
                csrRowPtr=self.d_mCurl_t.dPtr, csrColInd=self.d_mCurl_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(sCell)
            return sCell

        else:
            return self.mCurl_t.dot(vEdge)
        

    def discrete_laplace_v(self, sCell):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        if c.use_gpu:
            assert len(sCell) == self.d_mLaplace_v.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mLaplace_v.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mLaplace_v.shape[0], \
                n=self.d_mLaplace_v.shape[1], nnz=self.d_mLaplace_v.nnz, alpha=1.0, \
                descr=self.d_mLaplace_v.cuSparseDescr, csrVal=self.d_mLaplace_v.dData, \
                csrRowPtr=self.d_mLaplace_v.dPtr, csrColInd=self.d_mLaplace_v.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mLaplace_v.dot(sCell)


    def discrete_laplace_t(self, sVertex):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        if c.use_gpu:
            assert len(sVertex) == self.d_mLaplace_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mLaplace_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mLaplace_t.shape[0], \
                n=self.d_mLaplace_t.shape[1], nnz=self.d_mLaplace_t.nnz, alpha=1.0, \
                descr=self.d_mLaplace_t.cuSparseDescr, csrVal=self.d_mLaplace_t.dData, \
                csrRowPtr=self.d_mLaplace_t.dPtr, csrColInd=self.d_mLaplace_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mLaplace_t.dot(sVertex)

        
    # The discrete gradient operator along the normal direction
    def discrete_grad_n(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mGrad_n.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mGrad_n.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_n.shape[0], \
                n=self.d_mGrad_n.shape[1], nnz=self.d_mGrad_n.nnz, alpha=1.0, \
                descr=self.d_mGrad_n.cuSparseDescr, csrVal=self.d_mGrad_n.dData, \
                csrRowPtr=self.d_mGrad_n.dPtr, csrColInd=self.d_mGrad_n.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_n.dot(sCell)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_grad_td(self, sVertex):
        '''With implied Dirichlet BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mGrad_td.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mGrad_td.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_td.shape[0], \
                n=self.d_mGrad_td.shape[1], nnz=self.d_mGrad_td.nnz, alpha=1.0, \
                descr=self.d_mGrad_td.cuSparseDescr, csrVal=self.d_mGrad_td.dData, \
                csrRowPtr=self.d_mGrad_td.dPtr, csrColInd=self.d_mGrad_td.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_td.dot(sVertex)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Neumann BC's
    def discrete_grad_tn(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mGrad_tn.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mGrad_tn.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mGrad_tn.shape[0], \
                n=self.d_mGrad_tn.shape[1], nnz=self.d_mGrad_tn.nnz, alpha=1.0, \
                descr=self.d_mGrad_tn.cuSparseDescr, csrVal=self.d_mGrad_tn.dData, \
                csrRowPtr=self.d_mGrad_tn.dPtr, csrColInd=self.d_mGrad_tn.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mGrad_tn.dot(sVertex)


    # The discrete skew gradient operator along the normal direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_skewgrad_nd(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mSkewgrad_nd.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mSkewgrad_nd.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mSkewgrad_nd.shape[0], \
                n=self.d_mSkewgrad_nd.shape[1], nnz=self.d_mSkewgrad_nd.nnz, alpha=1.0, \
                descr=self.d_mSkewgrad_nd.cuSparseDescr, csrVal=self.d_mSkewgrad_nd.dData, \
                csrRowPtr=self.d_mSkewgrad_nd.dPtr, csrColInd=self.d_mSkewgrad_nd.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mSkewgrad_nd.dot(sVertex)

    # The discrete skew gradient operator along the tangential direction
    def discrete_skewgrad_t(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mSkewgrad_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mSkewgrad_t.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mSkewgrad_t.shape[0], \
                n=self.d_mSkewgrad_t.shape[1], nnz=self.d_mSkewgrad_t.nnz, alpha=1.0, \
                descr=self.d_mSkewgrad_t.cuSparseDescr, csrVal=self.d_mSkewgrad_t.dData, \
                csrRowPtr=self.d_mSkewgrad_t.dPtr, csrColInd=self.d_mSkewgrad_t.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mSkewgrad_t.dot(sCell)
        

    def cell2vertex(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mCell2vertex.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mCell2vertex.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCell2vertex.shape[0], \
                n=self.d_mCell2vertex.shape[1], nnz=self.d_mCell2vertex.nnz, alpha=1.0, \
                descr=self.d_mCell2vertex.cuSparseDescr, csrVal=self.d_mCell2vertex.dData, \
                csrRowPtr=self.d_mCell2vertex.dPtr, csrColInd=self.d_mCell2vertex.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mCell2vertex.dot(sCell)


    def vertex2cell(self, sVertex):

        if c.use_gpu:
            assert len(sVertex) == self.d_mVertex2cell.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mVertex2cell.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mVertex2cell.shape[0], \
                n=self.d_mVertex2cell.shape[1], nnz=self.d_mVertex2cell.nnz, alpha=1.0, \
                descr=self.d_mVertex2cell.cuSparseDescr, csrVal=self.d_mVertex2cell.dData, \
                csrRowPtr=self.d_mVertex2cell.dPtr, csrColInd=self.d_mVertex2cell.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mVertex2cell.dot(sVertex)
        
    def cell2edge(self, sCell):

        if c.use_gpu:
            assert len(sCell) == self.d_mCell2edge.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sCell)

            vOut = np.zeros(self.d_mCell2edge.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mCell2edge.shape[0], \
                n=self.d_mCell2edge.shape[1], nnz=self.d_mCell2edge.nnz, alpha=1.0, \
                descr=self.d_mCell2edge.cuSparseDescr, csrVal=self.d_mCell2edge.dData, \
                csrRowPtr=self.d_mCell2edge.dPtr, csrColInd=self.d_mCell2edge.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mCell2edge.dot(sCell)

    def edge2cell(self, sEdge):

        if c.use_gpu:
            assert len(sEdge) == self.d_mEdge2cell.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sEdge)

            vOut = np.zeros(self.d_mEdge2cell.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mEdge2cell.shape[0], \
                n=self.d_mEdge2cell.shape[1], nnz=self.d_mEdge2cell.nnz, alpha=1.0, \
                descr=self.d_mEdge2cell.cuSparseDescr, csrVal=self.d_mEdge2cell.dData, \
                csrRowPtr=self.d_mEdge2cell.dPtr, csrColInd=self.d_mEdge2cell.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mEdge2cell.dot(sEdge)
