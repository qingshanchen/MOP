import numpy as np
import scipy
#import cupy as cp
#import cupyx
#from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
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
        self.use_gpu = c.use_gpu # TODO - don't think we will need this

        # load appropriate module for defining objects on CPU or GPU
        if c.use_gpu:
            import cupy as xp
            from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix
            
            # copy grid areaCell and areaTriangle to local variables on CPU;
            # these are used by swe routines during init
            areaCell_cpu = g.areaCell.get()
            areaTriangle_cpu = g.areaTriangle.get()
        else:
            import numpy as xp
            from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

            areaCell_cpu = g.areaCell
            areaTriangle_cpu = g.areaTriangle

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
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nEdges))
        self.mDiv_v = A.tocsr( )

        
        #
        # Divergence on dual (triangle)
        #
        ## Construct the matrix representing the discrete div on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div_trig(g.verticesOnEdge, g.dcEdge, areaTriangle_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nVertices, g.nEdges))
        self.mDiv_t = A.tocsr( )
        

        #
        # Curl on primal
        #
        ## Construct the matrix representing the discrete curl on the primal mesh (Voronoi cells)
        ## No-slip BCs assumed on the boundary.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl(g.cellsOnEdge, g.dvEdge, areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nEdges))
        self.mCurl_v = A.tocsr( )


        #
        # Curl on dual
        #
        ## Construct the matrix representing the discrete curl on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, areaTriangle_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nVertices, g.nEdges))
        self.mCurl_t = A.tocsr( )


        #
        # Laplace on primal (voronoi)
        #
        ## Construct the matrix representing the discrete Laplace operator the primal
        ## mesh (Voronoi mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace(g.cellsOnEdge, g.dcEdge, g.dvEdge, \
                                                  areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nCells))
        self.mLaplace_v = A.tocsr( )


        #
        # Laplace on dual (triangle)
        #
        ## Construct the matrix representing the discrete Laplace operator the dual
        ## mesh (triangular mesh). Homogeneous Neuman BC's are assumed.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_laplace_triangle_neumann(g.boundaryEdgeMark, \
                                    g.verticesOnEdge, g.dvEdge, g.dcEdge, areaTriangle_cpu)
                                                  
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                xp.asarray(cols[:nEntries]))), shape=(g.nVertices, g.nVertices))
        self.mLaplace_t = A.tocsr( )


        #
        # Gradient normal
        #
        ## Construct the matrix representing the discrete grad operator along the normal direction.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nCells))
        self.mGrad_n = A.tocsr( )
        
        # ver 8.6.0 of cupyx does not have "tolil()" implemented; have to work around
        if c.use_gpu:
            A = A.get()
            
        A_n = A.tolil()   
        A_n[:,0] = 0.
        #A[:,0] = 0. try this instead of constructing A_n; change next lines as well.
        self.mGrad_n_n = A_n.tocsr( )
        self.mGrad_n_n.eliminate_zeros()

        if c.use_gpu:
            self.mGrad_n_n = csr_matrix(self.mGrad_n_n)


        #
        # Gradient tangential(?) with Dirichlet
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Dirichlet BC's 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_td(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nVertices))
        self.mGrad_td = A.tocsr( )


        #
        # Gradient tangential(?) with Neumann
        #
        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Neumann BC's
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_tn(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nVertices))
        self.mGrad_tn = A.tocsr( )


        #
        # Skew gradient tangential
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction.  mSkewgrad_t = mGrad_n
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_t(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_t = A.tocsr( )


        #
        # Skew gradient tangential w. Dirichlet
        #
        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction. Homogeneous Dirichlet assumed
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_td(g.cellsOnEdge, g.dcEdge, g.boundaryCellMark)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nCells))
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
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nVertices))
        self.mSkewgrad_nd = A.tocsr( )


        #
        # Map from cell to vertex
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, areaTriangle_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nVertices, g.nCells))
        self.mCell2vertex = A.tocsr( )


        # ver 8.6.0 of cupyx does not have "tolil()" implemented; have to work around
        if c.use_gpu:
            A = A.get()
            
        A_n = A.tolil( )
        A_n[:,0] = 0.       # zero for entry 0; Neumann
        self.mCell2vertex_n = A_n.tocsr()
        self.mCell2vertex_n.eliminate_zeros( )

        if c.use_gpu:
            self.mCell2vertex_n = csr_matrix(self.mCell2vertex_n)
        
        #
        # Map cell to vertex w. Dirichlet
        #
        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh; homogeneous Dirichlet BC's are assumed
        ## On a global sphere, cell 0 is considered the single boundary pt.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex_psi(g.cellsOnVertex, g.kiteAreasOnVertex, areaTriangle_cpu, \
                                                 g.boundaryCellMark, c.on_a_global_sphere)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                               xp.asarray(cols[:nEntries]))), shape=(g.nVertices, g.nCells))
        self.mCell2vertex_psi = A.tocsr( )


        #
        # Map vertex to cell
        #
        ## Construct the matrix representing the mapping from the dual mesh onto the primal
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_vertex2cell(g.cellsOnVertex, g.kiteAreasOnVertex, areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nVertices))
        self.mVertex2cell = A.tocsr( )


        #
        # Map cell to edge
        #
        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )


        #
        # Map edge to cell
        # 
        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, areaCell_cpu)
        A = coo_matrix((xp.asarray(valEntries[:nEntries]),  (xp.asarray(rows[:nEntries]), \
                                    xp.asarray(cols[:nEntries]))), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        
        ## Some temporary variables as place holders
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)


            

            
    def discrete_div_v(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''
        if self.use_gpu:
            vEdge_d = cp.asarray(vEdge)
            result_d = self.mDiv_v.dot(vEdge_d)
            return result_d.get()
        else:
            return self.mDiv_v.dot(vEdge)
            

    def discrete_div_t(self, vEdge):
        '''
        No flux boundary conditions implied on the boundary.
        '''
        if self.use_gpu:
            vEdge_d = cp.asarray(vEdge)
            result_d = self.mDiv_t.dot(vEdge_d)
            return result_d.get()
        else:
            return self.mDiv_t.dot(vEdge)
        

    def discrete_curl_v(self, vEdge):
        '''
        The discrete curl operator on the primal mesh.
        No-slip boundary conditions implied on the boundary.
        '''
        if self.use_gpu:
            vEdge_d = cp.asarray(vEdge)
            result_d = self.mCurl_v.dot(vEdge_d)
            return result_d.get()
        else:
            return self.mCurl_v.dot(vEdge)


    def discrete_curl_t(self, vEdge):
        '''
        The discrete curl operator on the dual mesh.
        '''
        if self.use_gpu:
            vEdge_d = cp.asarray(vEdge)
            result_d = self.mCurl_t.dot(vEdge_d)
            return result_d.get()
        else:
            return self.mCurl_t.dot(vEdge)
        

    def discrete_laplace_v(self, sCell):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''
        if self.use_gpu:
            sCell_d = cp.asarray(sCell)
            result_d = self.mLaplace_v.dot(sCell_d)
            return result_d.get()
        else:
            return self.mLaplace_v.dot(sCell)


    def discrete_laplace_t(self, sVertex):
        '''
        Homogeneous Neumann BC's implied on the boundary.
        '''
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = self.mLaplace_t.dot(sVertex_d)
            return result_d.get()
        else:
            return self.mLaplace_t.dot(sVertex)

        
    # The discrete gradient operator along the normal direction
    def discrete_grad_n(self, sCell):
        if self.use_gpu:
            sCell_d = cp.asarray(sCell)
            result_d = self.mGrad_n.dot(sCell_d)
            return result_d.get()
        else:
            return self.mGrad_n.dot(sCell)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_grad_td(self, sVertex):
        '''With implied Dirichlet BC's'''
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = self.mGrad_td.dot(sVertex_d)
            return result_d.get()
        else:
            return self.mGrad_td.dot(sVertex)


    # The discrete gradient operator along the tangential direction, assuming
    # homogeneous Neumann BC's
    def discrete_grad_tn(self, sVertex):
        '''With implied Neumann BC's'''
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = self.mGrad_tn.dot(sVertex_d)
            return result_d.get()
        else:
            return self.mGrad_tn.dot(sVertex)


    # The discrete skew gradient operator along the normal direction, assuming
    # homogeneous Dirichlet BC's
    def discrete_skewgrad_nd(self, sVertex):
        '''With implied Dirichlet BC's'''
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = self.mSkewgrad_nd.dot(sVertex_d)
            return result_d.get()
        else:
            return self.mSkewgrad_nd.dot(sVertex)


    # The discrete skew gradient operator along the normal direction, assuming
    # homogeneous Neumann BC's
    def discrete_skewgrad_nn(self, sVertex):
        '''With implied Neumann BC's.

           Since skew grad in the normal direction is the opposite of grad 
           in the tangent direction, we re-use the coefficient matrix of the latter, 
           but add an negative sign. See (B.4) and (B.6) of CJT21.
        '''
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = -self.mGrad_tn.dot(sVertex_d)
            return result_d.get()
        else:
            return -self.mGrad_tn.dot(sVertex)

    
    # The discrete skew gradient operator along the tangential direction
    def discrete_skewgrad_t(self, sCell):
        if self.use_gpu:
            sCell_d = cp.asarray(sCell)
            result_d = self.mSkewgrad_t.dot(sCell_d)
            return result_d.get()
        else:
            return self.mSkewgrad_t.dot(sCell)
        

    def cell2vertex(self, sCell):
        if self.use_gpu:
            sCell_d = cp.asarray(sCell)
            result_d = self.mCell2vertex.dot(sCell_d)
            return result_d.get()
        else:
            return self.mCell2vertex.dot(sCell)


    def vertex2cell(self, sVertex):
        if self.use_gpu:
            sVertex_d = cp.asarray(sVertex)
            result_d = self.mVertex2cell.dot(sVertex_d)
            return result_d.get()
        else:
            return self.mVertex2cell.dot(sVertex)

    
    def cell2edge(self, sCell):
        if self.use_gpu:
            sCell_d = cp.asarray(sCell)
            result_d = self.mCell2edge.dot(sCell_d)
            return result_d.get()
        else:
            return self.mCell2edge.dot(sCell)

    
    def edge2cell(self, sEdge):
        if self.use_gpu:
            sEdge_d = cp.asarray(sEdge)
            result_d = self.mEdge2cell.dot(sEdge_d)
            return result_d.get()
        else:
            return self.mEdge2cell.dot(sEdge)
