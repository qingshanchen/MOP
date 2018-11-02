import numpy as np
import Parameters as c
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye, diags, bmat
from scipy.sparse.linalg import spsolve, splu, factorized
from swe_comp import swe_comp as cmp

class EllipticCPL:
    def __init__(self, A, linear_solver, env):


        if linear_solver is 'lu':
            self.A = A.tocsc( )
            
        elif linear_solver is 'amgx':
            import pyamgx

            pyamgx.initialize( )

            hA = A.tocsr( )
            AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/FGMRES_AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/AMG_AGGREGATION_CG.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PBICGSTAB_AGGREGATION_W_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/AGGREGATION_JACOBI.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS.json'
            #AMGX_CONFIG_FILE_NAME = 'amgx_config/PCGF_CLASSICAL_AGGRESSIVE_PMIS_JACOBI.json'
 
            cfg = pyamgx.Config( ).create_from_file(AMGX_CONFIG_FILE_NAME) 
            rsc = pyamgx.Resources().create_simple(cfg)
            mode = 'dDDI'

            # Create solver:
            self.amgx = pyamgx.Solver().create(rsc, cfg, mode)

            # Create matrices and vectors:
            self.d_A = pyamgx.Matrix().create(rsc, mode)
            self.d_x = pyamgx.Vector().create(rsc, mode)
            self.d_b = pyamgx.Vector().create(rsc, mode)

            self.d_A.upload_CSR(hA)

            # Setup and solve system:
            # self.amgx.setup(d_A)

            ## Clean up:
            #A.destroy()
            #x.destroy()
            #b.destroy()
            #self.amgx.destroy()
            #rsc.destroy()
            #cfg.destroy()

            #pyamgx.finalize()
        else:
            raise ValueError("Invalid solver choice.")

    def solve(self, A, b, x, env=None, linear_solver='lu'):
        
        if linear_solver is 'lu':
            x[:] = spsolve(A, b)

        elif linear_solver is 'amgx':
            self.d_b.upload(b)
            self.d_x.upload(x)
            #self.d_A.replace_coefficients(A.data)
            self.d_A.upload_CSR(A)
            self.amgx.setup(self.d_A)
            self.amgx.solve(self.d_b, self.d_x)
            self.d_x.download(x)
        else:
            raise ValueError("Invalid solver choice.")

        
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

        self.max_iter = c.max_iter
        self.err_tol = c.err_tol

        self.areaCell = g.areaCell.copy()
        self.areaTriangle = g.areaTriangle.copy()

        self.on_a_global_sphere = c.on_a_global_sphere

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

        ## Construct the matrix representing the discrete div on the primal mesh (Voronoi cells)
        ## No-flux BCs assumed on the boundary
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mDiv_v = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_v = Device_CSR(self.mDiv_v, env)

        ## Construct the matrix representing the discrete div on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_div_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mDiv_t = A.tocsr( )

        if c.use_gpu:
            self.d_mDiv_t = Device_CSR(self.mDiv_t, env)
            
        ## Construct the matrix representing the discrete curl on the primal mesh (Voronoi cells)
        ## No-slip BCs assumed on the boundary.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl(g.cellsOnEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mCurl_v = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_v = Device_CSR(self.mCurl_v, env)

        ## Construct the matrix representing the discrete curl on the dual mesh (triangles)
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_curl_trig(g.verticesOnEdge, g.dcEdge, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nEdges))
        self.mCurl_t = A.tocsr( )

        if c.use_gpu:
            self.d_mCurl_t = Device_CSR(self.mCurl_t, env)
            
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

        ## Construct the matrix representing the discrete grad operator along the normal direction.
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mGrad_n = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_n = Device_CSR(self.mGrad_n, env)

        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Dirichlet BC's 
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_td(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_td = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_td = Device_CSR(self.mGrad_td, env)

        ## Construct the matrix representing the discrete grad operator for the dual
        ## mesh, with implied homogeneous Neumann BC's
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_grad_tn(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mGrad_tn = A.tocsr( )

        if c.use_gpu:
            self.d_mGrad_tn = Device_CSR(self.mGrad_tn, env)

        ## Construct the matrix representing the discrete skew grad operator 
        ## along the tangential direction.  mSkewgrad_t = mGrad_n
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_n(g.cellsOnEdge, g.dcEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mSkewgrad_t = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_t = Device_CSR(self.mSkewgrad_t, env)

        ## Construct the matrix representing the discrete skew grad operator 
        ## along the normal direction. Homogeneous Dirichlet assumed.
        ## mSkewgrad_n = - mGrad_td
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_discrete_skewgrad_t(g.verticesOnEdge, g.dvEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nVertices))
        self.mSkewgrad_n = A.tocsr( )

        if c.use_gpu:
            self.d_mSkewgrad_n = Device_CSR(self.mSkewgrad_n, env)

        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex = Device_CSR(self.mCell2vertex, env)

        ## Construct the matrix representing the mapping from the primary mesh onto the dual
        ## mesh; homogeneous Dirichlet BC's are assumed
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2vertex_psi(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaTriangle, g.boundaryCellMark, c.on_a_global_sphere)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nVertices, g.nCells))
        self.mCell2vertex_psi = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2vertex_psi = Device_CSR(self.mCell2vertex_psi, env)
            
        ## Construct the matrix representing the mapping from the dual mesh onto the primal
        ## mesh
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_vertex2cell(g.cellsOnVertex, g.kiteAreasOnVertex, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nVertices))
        self.mVertex2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mVertex2cell = Device_CSR(self.mVertex2cell, env)

        ## Construct the matrix representing the mapping from cells to edges
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_cell2edge(g.cellsOnEdge)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nEdges, g.nCells))
        self.mCell2edge = A.tocsr( )

        if c.use_gpu:
            self.d_mCell2edge = Device_CSR(self.mCell2edge, env)

        ## Construct the matrix representing the mapping from edges to cells
        nEntries, rows, cols, valEntries = \
            cmp.construct_matrix_edge2cell(g.cellsOnEdge, g.dcEdge, g.dvEdge, g.areaCell)
        A = coo_matrix((valEntries[:nEntries],  (rows[:nEntries], \
                               cols[:nEntries])), shape=(g.nCells, g.nEdges))
        self.mEdge2cell = A.tocsr( )

        if c.use_gpu:
            self.d_mEdge2cell = Device_CSR(self.mEdge2cell, env)

        self.mThicknessInv = eye(g.nEdges)   # This is only a space holder
#        if c.use_gpu:                        # Need to update at every step
#            d_mThicknessInv = Device_CSR(self.mThicknessInv.to_csr(), env)

        if c.component_for_hamiltonian == 'normal':
            self.leftM, self.rightM, self.coefM = self.construct_EllipticCPL_coefM_n(env, g, c)
        elif c.component_for_hamiltonian == 'tangential':
            self.leftM, self.rightM, self.coefM = self.construct_EllipticCPL_coefM_t(env, g, c)
        elif c.component_for_hamiltonian == 'normal_tangent':
            self.leftM_n, self.rightM_n, self.leftM_t, self.rightM_t, self.coefM = \
                                                self.construct_EllipticCPL_coefM_nt(env, g, c)
        else:
            raise ValueError("Invalid value for component_for_hamiltonian")
        
        self.POcpl = EllipticCPL(self.coefM, c.linear_solver, env)
            
        self.scalar_cell = np.zeros(g.nCells)
        self.scalar_vertex = np.zeros(g.nVertices)
        if not c.on_a_global_sphere:
            self.scalar_cell_interior = np.zeros(nCellsInterior)

    def construct_EllipticCPL_coefM_t(self, env, g, c):
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell, 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[self.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            mAreaCell_psi.eliminate_zeros( )
            
#        if c.use_gpu:
#            self.d_mAreaCell = Device_CSR(mAreaCell, env)

        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi
        AC = mAreaCell_psi * self.mCurl_v
        AMD = mAreaCell_phi * self.mVertex2cell * self.mDiv_t
        GN = self.mGrad_tn * self.mCell2vertex
        
        leftM = bmat([[AC],[AMD]], format='csr')
        rightM = bmat([[self.mSkewgrad_t, GN]], format='csr')

        leftM.eliminate_zeros( )
        rightM.eliminate_zeros( )

        
        thickness_edge = np.zeros(g.nEdges)
        thickness_edge[:] = 1000.    # Any non-zero should suffice
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        coefM = leftM * self.mThicknessInv * rightM

        if c.on_a_global_sphere:
            coefM[0,0] = 1.
            coefM[g.nCells, g.nCells] = 1.
        else:
            coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            coefM[g.nCells, g.nCells] = 1.

        return leftM, rightM, coefM


    def construct_EllipticCPL_coefM_n(self, env, g, c):
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell, 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[self.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            mAreaCell_psi.eliminate_zeros( )
            
#        if c.use_gpu:
#            self.d_mAreaCell = Device_CSR(mAreaCell, env)

        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi
        AMC = mAreaCell_psi * self.mVertex2cell * self.mCurl_t
        AD = mAreaCell_phi * self.mDiv_v
        SN = self.mSkewgrad_n * self.mCell2vertex_psi
        
        leftM = bmat([[AMC],[AD]], format='csr')
        rightM = bmat([[SN, self.mGrad_n]], format='csr')

        #AC = mAreaCell_psi * self.mCurl_v
        #AMD = mAreaCell_phi * self.mVertex2cell * self.mDiv_t
        #GN = self.mGrad_tn * self.mCell2vertex
        
        #leftM = bmat([[AC],[AMD]], format='csr')
        #rightM = bmat([[self.mSkewgrad_t, GN]], format='csr')

        leftM.eliminate_zeros( )
        rightM.eliminate_zeros( )

        
        thickness_edge = np.zeros(g.nEdges)
        thickness_edge[:] = 1000.    # Any non-zero should suffice
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        coefM = leftM * self.mThicknessInv * rightM

        if c.on_a_global_sphere:
            coefM[0,0] = 1.
            coefM[g.nCells, g.nCells] = 1.
        else:
            coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            coefM[g.nCells, g.nCells] = 1.

        return leftM, rightM, coefM
    

    def construct_EllipticCPL_coefM_nt(self, env, g, c):
        # A diagonal matrix representing scaling by cell areas
        mAreaCell = diags(g.areaCell, 0, format='csr')
        mAreaCell_phi = mAreaCell.copy( )
        mAreaCell_phi[0,0] = 0.
        mAreaCell_phi.eliminate_zeros( )

        if c.on_a_global_sphere:
            mAreaCell_psi = mAreaCell_phi.copy( )
        else:
            areaCell_psi = g.areaCell.copy( )
            areaCell_psi[self.cellBoundary - 1] = 0.
            mAreaCell_psi = diags(areaCell_psi, 0, format='csr')
            mAreaCell_psi.eliminate_zeros( )
            
        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi, using the normal vector
        AMC = mAreaCell_psi * self.mVertex2cell * self.mCurl_t
        AD = mAreaCell_phi * self.mDiv_v
        SN = self.mSkewgrad_n * self.mCell2vertex_psi
        
        leftM_n = bmat([[AMC],[AD]], format='csr')
        rightM_n = bmat([[SN, self.mGrad_n]], format='csr')
        leftM_n.eliminate_zeros( )
        rightM_n.eliminate_zeros( )

        ## Construct the coefficient matrix for the coupled elliptic
        ## system for psi and phi, using the tangential vecotrs
        AC = mAreaCell_psi * self.mCurl_v
        AMD = mAreaCell_phi * self.mVertex2cell * self.mDiv_t
        GN = self.mGrad_tn * self.mCell2vertex
        
        leftM_t = bmat([[AC],[AMD]], format='csr')
        rightM_t = bmat([[self.mSkewgrad_t, GN]], format='csr')

        leftM_t.eliminate_zeros( )
        rightM_t.eliminate_zeros( )

        ## Construct an artificial thickness vector
        thickness_edge = np.zeros(g.nEdges)
        thickness_edge[:] = 1000.    # Any non-zero should suffice
        self.mThicknessInv.data[0,:] = 1./thickness_edge
        
        coefM = 0.5 * leftM_n * self.mThicknessInv * rightM_n
        coefM += 0.5 * leftM_t * self.mThicknessInv * rightM_t

        if c.on_a_global_sphere:
            coefM[0,0] = 1.
            coefM[g.nCells, g.nCells] = 1.
        else:
            coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            coefM[g.nCells, g.nCells] = 1.

        return leftM_n, rightM_n, leftM_t, rightM_t, coefM
    
    
    def update_matrix_for_coupled_elliptic(self, thickness_edge, c, g):
        self.mThicknessInv.data[0,:] = 1./thickness_edge

        if c.component_for_hamiltonian == 'normal':
            self.coefM = self.leftM * self.mThicknessInv * self.rightM
        elif c.component_for_hamiltonian == 'tangential':
            self.coefM = self.leftM * self.mThicknessInv * self.rightM
        elif c.component_for_hamiltonian == 'normal_tangent':
            self.coefM = 0.5 * self.leftM_n * self.mThicknessInv * self.rightM_n
            self.coefM += 0.5 * self.leftM_t * self.mThicknessInv * self.rightM_t
        else:
            raise ValueError("Invalid value for component_for_hamiltonian")

        if c.on_a_global_sphere:
            self.coefM[0,0] = 1.
            self.coefM[g.nCells, g.nCells] = 1.
        else:
            self.coefM[self.cellBoundary-1, self.cellBoundary-1] = 1.
            self.coefM[g.nCells, g.nCells] = 1.

            
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
    def discrete_skewgrad_n(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mSkewgrad_n.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

            vOut = np.zeros(self.d_mSkewgrad_n.shape[0])
            d_vectorOut = self.env.cuda.to_device(vOut)
            self.env.cuSparse.csrmv(trans='N', m=self.d_mSkewgrad_n.shape[0], \
                n=self.d_mSkewgrad_n.shape[1], nnz=self.d_mSkewgrad_n.nnz, alpha=1.0, \
                descr=self.d_mSkewgrad_n.cuSparseDescr, csrVal=self.d_mSkewgrad_n.dData, \
                csrRowPtr=self.d_mSkewgrad_n.dPtr, csrColInd=self.d_mSkewgrad_n.dInd, \
                           x=d_vectorIn, beta=0., y=d_vectorOut)
            d_vectorOut.copy_to_host(vOut)
            return vOut

        else:
            return self.mSkewgrad_n.dot(sVertex)

    # The discrete skew gradient operator along the tangential direction
    def discrete_skewgrad_t(self, sVertex):
        '''With implied Neumann BC's'''

        if c.use_gpu:
            assert len(sVertex) == self.d_mSkewgrad_t.shape[1], \
                "Dimensions do not match."
            d_vectorIn = self.env.cuda.to_device(sVertex)

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
            return self.mSkewgrad_t.dot(sVertex)
        

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


        
        
        
        

        
