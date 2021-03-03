import numpy as np
import cupy as cp
import cupyx
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from swe_comp import swe_comp as cmp

class Device_CSR:
    def __init__(self, A):
        self.dData = env.cuda.to_device(A.data)
        self.dPtr = env.cuda.to_device(A.indptr)
        self.dInd = env.cuda.to_device(A.indices)
        self.shape = A.shape
        self.nnz = A.nnz
        self.cuSparseDescr = env.cuSparse.matdescr( )
#        self.d_vectOut = env.cuda.device_array


class VectorCalculus:
    def __init__(self, g, c):

        self.linear_solver = c.linear_solver
        self.max_iters = c.max_iters
        self.err_tol = c.err_tol
        self.use_gpu = c.use_gpu

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

        
        #
        # Divergence on dual (triangle)
        #
        ## Construct the matrix representing the discrete div on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mDiv_t = A.tocsr( )
        

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


        #
        # Curl on dual
        #
        ## Construct the matrix representing the discrete curl on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mCurl_t = A.tocsr( )


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


        #
        # Map cell to edge
        #
        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )


        #
        # Map edge to cell
        # 
        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        
        ## Some temporary variables as place holders
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)


        ## Move matrices to GPU
        if self.use_gpu:
            # TODO - do we need "self.use_gpu", or can we just use c.use_gpu
            self.mDiv_v = cupyx.scipy.sparse.csr_matrix(self.mDiv_v)
            self.mDiv_t = cupyx.scipy.sparse.csr_matrix(self.mDiv_t)
            self.mCurl_v = cupyx.scipy.sparse.csr_matrix(self.mCurl_v)
            self.mCurl_t = cupyx.scipy.sparse.csr_matrix(self.mCurl_t)
            self.mLaplace_v = cupyx.scipy.sparse.csr_matrix(self.mLaplace_v)
            self.mLaplace_t = cupyx.scipy.sparse.csr_matrix(self.mLaplace_t)
            self.mGrad_n = cupyx.scipy.sparse.csr_matrix(self.mGrad_n)
            self.mGrad_td = cupyx.scipy.sparse.csr_matrix(self.mGrad_td)
            self.mGrad_tn = cupyx.scipy.sparse.csr_matrix(self.mGrad_tn)
            self.mSkewgrad_t = cupyx.scipy.sparse.csr_matrix(self.mSkewgrad_t)
            self.mSkewgrad_td = cupyx.scipy.sparse.csr_matrix(self.mSkewgrad_td) # needed?
            self.mSkewgrad_nd = cupyx.scipy.sparse.csr_matrix(self.mSkewgrad_nd)
            self.mCell2vertex = cupyx.scipy.sparse.csr_matrix(self.mCell2vertex)
            self.mCell2vertex_n = cupyx.scipy.sparse.csr_matrix(self.mCell2vertex_n) # needed?
            self.mCell2vertex_psi = cupyx.scipy.sparse.csr_matrix(self.mCell2vertex_psi) # needed?
            self.mVertex2cell = cupyx.scipy.sparse.csr_matrix(self.mVertex2cell)
            self.mCell2edge = cupyx.scipy.sparse.csr_matrix(self.mCell2edge)
            self.mEdge2cell = cupyx.scipy.sparse.csr_matrix(self.mEdge2cell)

            
    def discrete_div_v(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        return self.mDiv_v.dot(vEdge)


    def discrete_div_t(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''

        return self.mDiv_t.dot(vEdge)
        

    def discrete_curl_v(self, vEdge):
        '''
        The discrete curl operator on the primal mesh.
        No-slip boundary conditions implied on the boundary.
        '''

        return self.mCurl_v.dot(vEdge)


    def discrete_curl_t(self, vEdge):
        '''
        The discrete curl operator on the dual mesh.
        '''

        return self.mCurl_t.dot(vEdge)
        

    def discrete_laplace_v(self, sCell):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        return self.mLaplace_v.dot(sCell)


    def discrete_laplace_t(self, sVertex):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''

        return self.mLaplace_t.dot(sVertex)

        
    # The discrete gradient operator along the normal direction
    def discrete_grad_n(self, sCell):

        return self.mGrad_n.dot(sCell)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_grad_td(self, sVertex):
        '''With implied Dirichlet BC's'''

        return self.mGrad_td.dot(sVertex)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Neumann BC's
    def discrete_grad_tn(self, sVertex):
        '''With implied Neumann BC's'''

        return self.mGrad_tn.dot(sVertex)


    # The discrete skew gradient operator along the normal direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_skewgrad_nd(self, sVertex):
        '''With implied Neumann BC's'''

        return self.mSkewgrad_nd.dot(sVertex)

    
    # The discrete skew gradient operator along the tangential direction
    def discrete_skewgrad_t(self, sCell):

        return self.mSkewgrad_t.dot(sCell)
        

    def cell2vertex(self, sCell):

        return self.mCell2vertex.dot(sCell)


    def vertex2cell(self, sVertex):

        return self.mVertex2cell.dot(sVertex)

    
    def cell2edge(self, sCell):

        return self.mCell2edge.dot(sCell)

    
    def edge2cell(self, sEdge):

        return self.mEdge2cell.dot(sEdge)
