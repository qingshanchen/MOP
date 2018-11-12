module swe_comp
implicit none

contains

subroutine construct_matrix_cell2vertex(nVertices, vertexDegree, &
     cellsOnVertex, kiteAreasOnVertex, areaTriangle,  &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nVertices, vertexDegree
  integer, intent(in) :: cellsOnVertex(0:nVertices-1, 0:vertexDegree-1)
  double precision, intent(in)  :: kiteAreasOnVertex(0:nVertices-1, 0:vertexDegree-1), &
       areaTriangle(0:nVertices-1)
  integer, intent(out)  :: nEntries, rows(0:nVertices*vertexDegree-1), cols(0:nVertices*vertexDegree-1)
  double precision, intent(out) :: valEntries(0:nVertices*vertexDegree-1)

  integer :: iVertex, iCell, i, iEntry

  iEntry = 0

  do iVertex = 0, nVertices-1
      do i = 0, vertexDegree-1
         iCell = cellsOnVertex(iVertex, i) - 1
         
!         scalar_vertex(iVertex) = scalar_vertex(iVertex) + &
!              kiteAreasOnVertex(iVertex, i)*scalar_cell(iCell)/areaTriangle(iVertex)

         rows(iEntry) = iVertex
         cols(iEntry) = iCell
         valEntries(iEntry) = kiteAreasOnVertex(iVertex, i) / areaTriangle(iVertex)
         iEntry = iEntry + 1
         
      end do
   end do
   nEntries = iEntry

end subroutine construct_matrix_cell2vertex


subroutine construct_matrix_cell2vertex_psi(nVertices, vertexDegree, nCells, &
     cellsOnVertex, kiteAreasOnVertex, areaTriangle, boundaryCellMark, onGlobalSphere, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nVertices, vertexDegree, nCells
  integer, intent(in) :: cellsOnVertex(0:nVertices-1, 0:vertexDegree-1), &
                         boundaryCellMark(0:nCells-1), onGlobalSphere
  double precision, intent(in)  :: kiteAreasOnVertex(0:nVertices-1, 0:vertexDegree-1), &
       areaTriangle(0:nVertices-1)
  integer, intent(out)  :: nEntries, rows(0:nVertices*vertexDegree-1), cols(0:nVertices*vertexDegree-1)
  double precision, intent(out) :: valEntries(0:nVertices*vertexDegree-1)

  integer :: iVertex, iCell, i, iEntry

  iEntry = 0

  do iVertex = 0, nVertices-1
      do i = 0, vertexDegree-1
         iCell = cellsOnVertex(iVertex, i) - 1

         if (onGlobalSphere == 1) then
            if (iCell > 0) then
                rows(iEntry) = iVertex
                cols(iEntry) = iCell
                valEntries(iEntry) = kiteAreasOnVertex(iVertex, i) / areaTriangle(iVertex)
                iEntry = iEntry + 1
             endif
         else
             if (boundaryCellMark(iCell) .EQ. 0) then
                rows(iEntry) = iVertex
                cols(iEntry) = iCell
                valEntries(iEntry) = kiteAreasOnVertex(iVertex, i) / areaTriangle(iVertex)
                iEntry = iEntry + 1
             end if
         endif
      end do
   end do
   nEntries = iEntry

end subroutine construct_matrix_cell2vertex_psi


subroutine construct_matrix_vertex2cell(nVertices, nCells, vertexDegree, &
     cellsOnVertex, kiteAreasOnVertex, areaCell,  &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nVertices, nCells, vertexDegree
  integer, intent(in) :: cellsOnVertex(0:nVertices-1, 0:vertexDegree-1)
  double precision, intent(in)  :: kiteAreasOnVertex(0:nVertices-1, 0:vertexDegree-1), &
       areaCell(0:nCells-1)
  integer, intent(out)  :: nEntries, rows(0:nVertices*vertexDegree-1), cols(0:nVertices*vertexDegree-1)
  double precision, intent(out) :: valEntries(0:nVertices*vertexDegree-1)

  integer :: iVertex, iCell, i, iEntry

  iEntry = 0

  do iVertex = 0, nVertices-1
      do i = 0, vertexDegree-1
         iCell = cellsOnVertex(iVertex, i) - 1
         
         rows(iEntry) = iCell
         cols(iEntry) = iVertex
         valEntries(iEntry) = kiteAreasOnVertex(iVertex, i) / areaCell(iCell)
         iEntry = iEntry + 1
         
      end do
   end do
   nEntries = iEntry

end subroutine construct_matrix_vertex2cell


subroutine construct_matrix_edge2cell(nCells, nEdges, &
     cellsOnEdge, dcEdge, dvEdge, areaCell, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nCells, nEdges
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: areaCell(0:nCells-1), dvEdge(0:nEdges-1), &
       dcEdge(0:nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)

  integer :: iEdge, cell0, cell1, iEntry
  double precision:: halfDiamond

  iEntry = 0
  
  do iEdge = 0, nEdges-1
     cell0 = cellsOnEdge(iEdge,0) - 1
     cell1 = cellsOnEdge(iEdge,1) - 1
     halfDiamond = 0.5 * 0.5 * dvEdge(iEdge) * dcEdge(iEdge)

     rows(iEntry) = cell0
     cols(iEntry) = iEdge
     valEntries(iEntry) = halfDiamond / areaCell(cell0)
     iEntry = iEntry + 1

     rows(iEntry) = cell1
     cols(iEntry) = iEdge
     valEntries(iEntry) = halfDiamond / areaCell(cell1)
     iEntry = iEntry + 1     
  end do

  nEntries = iEntry

end subroutine construct_matrix_edge2cell


subroutine construct_matrix_cell2edge(nEdges,  &
     cellsOnEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)

  integer :: iEdge, cell0, cell1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        rows(iEntry) = iEdge
        cols(iEntry) = cell0
        valEntries(iEntry) = 0.5
        iEntry = iEntry + 1

        rows(iEntry) = iEdge
        cols(iEntry) = cell1
        valEntries(iEntry) = 0.5
        iEntry = iEntry + 1
     end do

     nEntries = iEntry

end subroutine construct_matrix_cell2edge


! Construct the discrete divergence operator on the primal mesh
! The orientation on the edge is assumed to be from cell0 (first
! cell) to cell1 (second cell)
subroutine construct_matrix_discrete_div(nEdges, nCells, &
  cellsOnEdge, dvEdge, areaCell,  &
  nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: dvEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out)          :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1) 

  integer :: iEdge, cell0, cell1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
      cell0 = cellsOnEdge(iEdge,0) - 1
      cell1 = cellsOnEdge(iEdge,1) - 1

      rows(iEntry) = cell0
      cols(iEntry) = iEdge
      valEntries(iEntry) = dvEdge(iEdge) / areaCell(cell0)
      iEntry = iEntry + 1

      rows(iEntry) = cell1
      cols(iEntry) = iEdge
      valEntries(iEntry) = -dvEdge(iEdge) / areaCell(cell1)
      iEntry = iEntry + 1
        
   end do
   nEntries = iEntry

end subroutine construct_matrix_discrete_div


! Construct the discrete divergence operator on the dual mesh
! The orientation on the edge is assumed to be from vertex0 (first
! cell) to vertex1 (second cell)
subroutine construct_matrix_discrete_div_trig(nEdges, nVertices, &
  verticesOnEdge, dcEdge, areaTriangle,  &
  nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges, nVertices
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: dcEdge(0:nEdges-1), &
       areaTriangle(0:nVertices-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out)          :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1) 

  integer :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
      vertex0 = verticesOnEdge(iEdge,0) - 1
      vertex1 = verticesOnEdge(iEdge,1) - 1

      if (vertex0 .GE. 0) then 
          rows(iEntry) = vertex0
          cols(iEntry) = iEdge
          valEntries(iEntry) = dcEdge(iEdge) / areaTriangle(vertex0)
          iEntry = iEntry + 1
      end if

      if (vertex1 .GE. 0) then
          rows(iEntry) = vertex1
          cols(iEntry) = iEdge
          valEntries(iEntry) = -dcEdge(iEdge) / areaTriangle(vertex1)
          iEntry = iEntry + 1
      end if
        
   end do
   nEntries = iEntry

end subroutine construct_matrix_discrete_div_trig


! Construct the coefficient matrix representing the discrete curl
! operator on the primary mesh. 
! No slip boundary condition implied on the boundary.
subroutine construct_matrix_discrete_curl(nEdges, nCells,  &
     cellsOnEdge, dvEdge, areaCell, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dvEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  integer, intent(out)          :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)

  integer :: iEdge, cell0, cell1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        rows(iEntry) = cell0
        cols(iEntry) = iEdge
        valEntries(iEntry) = dvEdge(iEdge) / areaCell(cell0)
        iEntry = iEntry + 1

        rows(iEntry) = cell1
        cols(iEntry) = iEdge
        valEntries(iEntry) = -dvEdge(iEdge) / areaCell(cell1)
        iEntry = iEntry + 1
        
  end do

  nEntries = iEntry

  
end subroutine construct_matrix_discrete_curl


! Construct the coefficient matrix representing the discrete curl
! operator on the dual mesh. 
subroutine construct_matrix_discrete_curl_trig(nEdges, nVertices,  &
     verticesOnEdge, dcEdge, areaTriangle, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges, nVertices
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1), &
       areaTriangle(0:nVertices-1)
  integer, intent(out)          :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)

  integer :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1

        if (vertex0 .GE. 0) then
            rows(iEntry) = vertex0
            cols(iEntry) = iEdge
            valEntries(iEntry) = -dcEdge(iEdge) / areaTriangle(vertex0)
            iEntry = iEntry + 1
        end if

        if (vertex1 .GE. 0) then
            rows(iEntry) = vertex1
            cols(iEntry) = iEdge
            valEntries(iEntry) = dcEdge(iEdge) / areaTriangle(vertex1)
            iEntry = iEntry + 1
        end if
  end do

  nEntries = iEntry
  
end subroutine construct_matrix_discrete_curl_trig


! Homogeneous Neumann BC assumed on the boundary
subroutine construct_matrix_discrete_laplace(nEdges, nCells,  &
     cellsOnEdge, dcEdge, dvEdge, areaCell,  &
     nEntries, rows, cols, valEntries)
  
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1), dvEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  integer, intent(out)          :: nEntries, rows(0:4*nEdges-1), cols(0:4*nEdges-1)
  double precision, intent(out) :: valEntries(0:4*nEdges-1)

  integer:: iEdge, cell0, cell1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
     cell0 = cellsOnEdge(iEdge,0) - 1
     cell1 = cellsOnEdge(iEdge,1) - 1

     rows(iEntry) = cell0
     cols(iEntry) = cell0
     valEntries(iEntry) = -dvEdge(iEdge) / dcEdge(iEdge) / areaCell(cell0)
     iEntry = iEntry + 1

     rows(iEntry) = cell0
     cols(iEntry) = cell1
     valEntries(iEntry) = dvEdge(iEdge) / dcEdge(iEdge) / areaCell(cell0)
     iEntry = iEntry + 1
     
     rows(iEntry) = cell1
     cols(iEntry) = cell0
     valEntries(iEntry) = dvEdge(iEdge) / dcEdge(iEdge) / areaCell(cell1)
     iEntry = iEntry + 1

     rows(iEntry) = cell1
     cols(iEntry) = cell1
     valEntries(iEntry) = -dvEdge(iEdge) / dcEdge(iEdge) / areaCell(cell1)
     iEntry = iEntry + 1
  end do

  nEntries = iEntry
  
end subroutine construct_matrix_discrete_laplace


subroutine separate_boundary_interior_cells(nCells, nCellsInterior, nCellsBoundary, max_int, boundaryCell,  &
                                            cellInterior, cellBoundary, cellRankInterior)
  
  integer, intent(in)   :: nCells, nCellsInterior, nCellsBoundary, max_int
  integer, intent(in)   :: boundaryCell(0:nCells-1)
  integer, intent(out)  :: cellInterior(0:nCellsInterior-1), cellBoundary(0:nCellsBoundary-1), &
       &cellRankInterior(0:nCells-1)

  ! Local variables
  integer               :: index, index2, iCell


  index = 1
  index2 = 1


  do iCell = 0, nCells-1
     if (boundaryCell(iCell) .EQ. 0) then
        cellInterior(index-1) = iCell + 1
        cellRankInterior(iCell) = index
        index = index + 1
     else
        cellBoundary(index2-1) = iCell + 1
        cellRankInterior(iCell) = max_int
        index2 = index2 + 1
     end if
  end do

end subroutine separate_boundary_interior_cells

! cellBoundary: cells on the boundary
! cellInterior: cells not on the boundary
! cellRankInterior: the index of an interior cell in the sequence of all interior cells
! cellInner: cells that are not connected with boundary cells
! cellOuter: cells that are not inner cells
subroutine separate_boundary_interior_inner_cells(nCells, maxEdges, &
     nCellsInterior, nCellsBoundary, max_int, boundaryCell,  cellsOnCell, nEdgesOnCell, &
     cellInterior, cellBoundary, cellRankInterior, cellInner, cellOuter, &
     cellRankInner, nInnerCells, nOuterCells)

  
  integer, intent(in)   :: nCells, nCellsInterior, nCellsBoundary, max_int, maxEdges
  integer, intent(in)   :: boundaryCell(0:nCells-1), cellsOnCell(0:nCells-1, 0:maxEdges-1), &
       nEdgesOnCell(0:nCells-1)
  integer, intent(out)  :: cellInterior(0:nCellsInterior-1),  &
       cellBoundary(0:nCellsBoundary-1), cellRankInterior(0:nCells-1), &
       cellInner(0:nCells-1), cellOuter(0:nCells-1), cellRankInner(0:nCells-1), &
       nInnerCells, nOuterCells

  ! Local variables
  integer               :: index, index2, iCell, inner_indx, outer_indx

  index = 1
  index2 = 1

  inner_indx = 1
  outer_indx = 1


  do iCell = 0, nCells-1
     if (boundaryCell(iCell) .EQ. 0) then
        cellInterior(index-1) = iCell + 1
        cellRankInterior(iCell) = index
        index = index + 1

        ! To check for inner cells (not connected to boundary cells)
        if (sum(boundaryCell(cellsOnCell(iCell,0:nEdgesOnCell(iCell)-1)-1)) == 0) then
           cellInner(inner_indx-1) = iCell + 1
           cellRankInner(iCell) = inner_indx
           inner_indx = inner_indx + 1
        else
           cellOuter(outer_indx-1) = iCell + 1
           cellRankInner(iCell) = max_int
           outer_indx = outer_indx + 1
        end if
        
     else  
        cellBoundary(index2-1) = iCell + 1
        cellRankInterior(iCell) = max_int
        index2 = index2 + 1

        ! A boundary cell is automatically an outer cell
        cellOuter(outer_indx-1) = iCell + 1 
        cellRankInner(iCell) = max_int
        outer_indx = outer_indx + 1
        
     end if

     nInnerCells = inner_indx - 1
     nOuterCells = outer_indx - 1
     
  end do

end subroutine separate_boundary_interior_inner_cells


!! Homogeneous Neumann BC's assumed.
subroutine construct_matrix_discrete_laplace_triangle_neumann(nVertices, nEdges, &
     boundaryEdge, verticesOnEdge, dvEdge, dcEdge, areaTriangle, &
     nEntries, rows, cols, valEntries)

  integer, intent(in)    :: nVertices, nEdges
  integer, intent(in)    :: boundaryEdge(0:nEdges-1), verticesOnEdge(0:nEdges-1,0:1)
  double precision, intent(in)     :: dvEdge(0:nEdges-1), dcEdge(0:nEdges-1), areaTriangle(0:nVertices-1)
  integer, intent(out)   :: rows(0:4*nEdges-1), cols(0:4*nEdges-1), nEntries
  double precision, intent(out)    :: valEntries(0:4*nEdges-1)

  integer   :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
     vertex0 = verticesOnEdge(iEdge,0) - 1
     vertex1 = verticesOnEdge(iEdge,1) - 1
        
     if (boundaryEdge(iEdge) .EQ. 0) then
        rows(iEntry) = vertex0
        cols(iEntry) = vertex0
        valEntries(iEntry) = -dcEdge(iEdge)/dvEdge(iEdge)/areaTriangle(vertex0)
        iEntry = iEntry + 1

        rows(iEntry) = vertex0
        cols(iEntry) = vertex1
        valEntries(iEntry) = dcEdge(iEdge)/dvEdge(iEdge)/areaTriangle(vertex0)
        iEntry = iEntry + 1

        rows(iEntry) = vertex1
        cols(iEntry) = vertex0
        valEntries(iEntry) = dcEdge(iEdge)/dvEdge(iEdge)/areaTriangle(vertex1)
        iEntry = iEntry + 1

        rows(iEntry) = vertex1
        cols(iEntry) = vertex1
        valEntries(iEntry) = -dcEdge(iEdge)/dvEdge(iEdge)/areaTriangle(vertex1)
        iEntry = iEntry + 1
     end if
  end do
  
  nEntries = iEntry

end subroutine construct_matrix_discrete_laplace_triangle_neumann


subroutine separate_boundary_interior_edges(nEdges, &
     nEdgesInterior, nEdgesBoundary, max_int, boundaryEdgeMark,  &
     edgeInterior, edgeBoundary, edgeRankInterior, edgeRankBoundary)
  
  integer, intent(in)   :: nEdges, nEdgesInterior, nEdgesBoundary, max_int
  integer, intent(in)   :: boundaryEdgeMark(0:nEdges-1)
  integer, intent(out)  :: edgeInterior(0:nEdgesInterior-1), edgeBoundary(0:nEdgesBoundary-1), &
       edgeRankInterior(0:nEdges-1), edgeRankBoundary(0:nEdges-1)

  ! Local variables
  integer               :: index, index2, iEdge


  index = 1
  index2 = 1


  do iEdge = 0, nEdges-1
     if (boundaryEdgeMark(iEdge) .EQ. 0) then
        edgeInterior(index-1) = iEdge + 1
        edgeRankInterior(iEdge) = index
        edgeRankBoundary(iEdge) = max_int
        index = index + 1
     else
        edgeBoundary(index2-1) = iEdge + 1
        edgeRankInterior(iEdge) = max_int
        index2 = index2 + 1
     end if
  end do

end subroutine separate_boundary_interior_edges


! Construct the discrete gradient operator on the primal mesh
subroutine construct_matrix_discrete_grad_n(nEdges, &
     cellsOnEdge, dcEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

  integer :: iEdge, cell0, cell1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        
        rows(iEntry) = iEdge
        cols(iEntry) = cell0
        valEntries(iEntry) = -1./dcEdge(iEdge)
        iEntry = iEntry + 1

        rows(iEntry) = iEdge
        cols(iEntry) = cell1
        valEntries(iEntry) = 1./dcEdge(iEdge)
        iEntry = iEntry + 1
   end do

   nEntries = iEntry

end subroutine construct_matrix_discrete_grad_n


subroutine construct_matrix_discrete_grad_td(nEdges, &
     verticesOnEdge, dvEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dvEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

  integer :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1

        if (vertex0 .GE. 0 .and. vertex1 .GE. 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex0
            valEntries(iEntry) = -1./dvEdge(iEdge)
            iEntry = iEntry + 1

            rows(iEntry) = iEdge
            cols(iEntry) = vertex1
            valEntries(iEntry) = 1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else if (vertex0 .GE. 0 .and. vertex1 < 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex0
            valEntries(iEntry) = -1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else if (vertex1 .GE. 0 .and. vertex0 < 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex1
            valEntries(iEntry) = 1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else
           write(*,*) "Vertex indices in verticesOnEdge are wrong in discrete_grad_t. Exit."
           stop
        end if
   end do

   nEntries = iEntry

end subroutine construct_matrix_discrete_grad_td


subroutine construct_matrix_discrete_grad_tn(nEdges, &
     verticesOnEdge, dvEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dvEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

  integer :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1

        if (vertex0 .GE. 0 .and. vertex1 .GE. 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex0
            valEntries(iEntry) = -1./dvEdge(iEdge)
            iEntry = iEntry + 1

            rows(iEntry) = iEdge
            cols(iEntry) = vertex1
            valEntries(iEntry) = 1./dvEdge(iEdge)
            iEntry = iEntry + 1
        end if
   end do

   nEntries = iEntry
end subroutine construct_matrix_discrete_grad_tn

! Construct the discrete skew gradient operator on the dual mesh;
! the resulting vector is along the normal direction.
! Homogeneous Dirichlet BC's assumed 
subroutine construct_matrix_discrete_skewgrad_t(nEdges, &
     verticesOnEdge, dvEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dvEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

  integer :: iEdge, vertex0, vertex1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1

        if (vertex0 .GE. 0 .and. vertex1 .GE. 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex0
            valEntries(iEntry) = 1./dvEdge(iEdge)
            iEntry = iEntry + 1

            rows(iEntry) = iEdge
            cols(iEntry) = vertex1
            valEntries(iEntry) = -1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else if (vertex0 .GE. 0 .and. vertex1 < 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex0
            valEntries(iEntry) = 1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else if (vertex1 .GE. 0 .and. vertex0 < 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = vertex1
            valEntries(iEntry) = -1./dvEdge(iEdge)
            iEntry = iEntry + 1
         else
           write(*,*) "Vertex indices in verticesOnEdge are wrong in discrete_grad_t. Exit."
           stop
        end if
   end do

   nEntries = iEntry

end subroutine construct_matrix_discrete_skewgrad_t


! Construct the discrete skew gradient operator on the primal mesh;
! the resulting vector is along the tangential direction.
subroutine construct_matrix_discrete_skewgrad_n(nEdges, &
     cellsOnEdge, dcEdge, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

  integer :: iEdge, cell0, cell1, iEntry

  iEntry = 0
  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        rows(iEntry) = iEdge
        cols(iEntry) = cell0
        valEntries(iEntry) = -1./dcEdge(iEdge)
        iEntry = iEntry + 1

        rows(iEntry) = iEdge
        cols(iEntry) = cell1
        valEntries(iEntry) = 1./dcEdge(iEdge)
        iEntry = iEntry + 1
   end do

   nEntries = iEntry

end subroutine construct_matrix_discrete_skewgrad_n

end module
