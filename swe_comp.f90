module swe_comp
implicit none

contains

  subroutine separate_boundary_interior_cells(nCells, &
       nCellsInterior, nCellsBoundary, max_int, boundaryCell,  &
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

! Single out all the boundary cells and list them
! in the counter-clock wise order.
! ONLY works when there are no holes within the domain
subroutine boundary_cells_ordered(nCells, maxEdges, &
       nCellsBoundary, boundaryCellMark, cellsOnCell, &
       cellBoundary)
  
  integer, intent(in)   :: nCells, nCellsBoundary, maxEdges
  integer, intent(in)   :: boundaryCellMark(0:nCells-1), cellsOnCell(0:nCells-1,0:maxEdges-1)
  integer, intent(out)  :: cellBoundary(0:nCellsBoundary)

  ! Local variables
  integer               :: iCell, cell0, cell1, cell2, index

  index = 0

  ! Locate the first boundary cell
  do iCell = 0, nCells-1
     if (boundaryCellMark(iCell) > 0) then
        cell0 = iCell + 1
        cellBoundary(index) = cell0
        index = index + 1
        exit
     end if
  end do

  cell1 = cell0
  do
     cell2 = cellsOnCell(cell1-1,0)
     cellBoundary(index) = cell2
     index = index + 1

     ! Debugging
!     write(*,*) "cell1, cell2 = ", cell1, cell2
     ! End debugging
     
     if (cell2 == cell0) then
        exit
     else
        cell1 = cell2
     end if

     if (index > nCells) then
        write(*,*) "Endless loop!"
        stop
     end if
  end do

  if (index .NE. nCellsBoundary+1) then
     write(*,*) "Number of boundary cells not correct."
     write(*,*) "index = ", index-1
     write(*,*) "nCellsBoundary = ", nCellsBoundary
     stop
  end if

end subroutine boundary_cells_ordered


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

! Similar to the above, but for Poisson BVP on the primary
! mesh.
subroutine construct_discrete_laplace_neumann(nCells, nEdges, &
     cellsOnEdge, dvEdge, dcEdge, areaCell, nEntries, &
     rows, cols, valEntries)

  integer, intent(in)              :: nCells, nEdges
  integer, intent(in)              :: cellsOnEdge(0:nEdges-1,0:1)
  double precision, intent(in)     :: dvEdge(0:nEdges-1), dcEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  integer, intent(out)             :: rows(0:4*nEdges+nCells-1), cols(0:4*nEdges+nCells-1), &
       nEntries
  double precision, intent(out)    :: valEntries(0:4*nEdges+nCells-1)

  integer   :: iEdge, iCell1, iCell2, iEntry

  iEntry = 0

  do iEdge = 0, nEdges-1
      iCell1 = cellsOnEdge(iEdge,0) - 1
      iCell2 = cellsOnEdge(iEdge,1) - 1

      rows(iEntry) = iCell1
      cols(iEntry) = iCell1
      valEntries(iEntry) = -dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell1)
      iEntry = iEntry + 1

      rows(iEntry) = iCell1
      cols(iEntry) = iCell2
      valEntries(iEntry) = dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell1)
      iEntry = iEntry + 1
        
      rows(iEntry) = iCell2
      cols(iEntry) = iCell1
      valEntries(iEntry) = dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell2)
      iEntry = iEntry + 1

      rows(iEntry) = iCell2
      cols(iEntry) = iCell2
      valEntries(iEntry) = -dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell2)
      iEntry = iEntry + 1
  end do

  nEntries = iEntry

  ! Set all entries on the first row to zero except the diagonal term
  do iEntry = 0, nEntries-1
     if (rows(iEntry) .EQ. 0 .AND. cols(iEntry) .NE. 0) then
     !if (rows(iEntry)*cols(iEntry) .EQ. 0 .AND. rows(iEntry)+cols(iEntry) .NE. 0) then
        valEntries(iEntry) = 0.
     else if (rows(iEntry) .NE. 0 .AND. cols(iEntry) .EQ. 0) then
        valEntries(iEntry) = 0.
     end if
  end do

end subroutine construct_discrete_laplace_neumann



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
subroutine construct_matrix_discrete_skewgrad_nd(nEdges, &
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

end subroutine construct_matrix_discrete_skewgrad_nd


! Construct the discrete skew gradient operator on the primal mesh;
! the resulting vector is along the tangential direction.
subroutine construct_matrix_discrete_skewgrad_t(nEdges, &
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

end subroutine construct_matrix_discrete_skewgrad_t


! Construct the discrete skew gradient operator on the primal mesh;
! the resulting vector is along the tangential direction.
! Homogeneous Dirichlet is assumed on the scalar variable
subroutine construct_matrix_discrete_skewgrad_td(nEdges, nCells, &
     cellsOnEdge, dcEdge, boundaryCellMark, &
     nEntries, rows, cols, valEntries)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1), boundaryCellMark(0:nCells-1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1)
  double precision, intent(out) :: valEntries(0:2*nEdges-1)
  integer, intent(out) :: nEntries, rows(0:2*nEdges-1), cols(0:2*nEdges-1)

   integer :: iEdge, cell0, cell1, iEntry

   iEntry = 0

   if (sum(boundaryCellMark) .EQ. 0) then ! Case of a global spehre
      do iEdge = 0, nEdges-1
         cell0 = cellsOnEdge(iEdge,0) - 1
         cell1 = cellsOnEdge(iEdge,1) - 1

         if (cell0 > 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = cell0
            valEntries(iEntry) = -1./dcEdge(iEdge)
            iEntry = iEntry + 1
         end if

         if (cell1 > 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = cell1
            valEntries(iEntry) = 1./dcEdge(iEdge)
            iEntry = iEntry + 1
         end if
     end do
   
   else 
      
      do iEdge = 0, nEdges-1
         cell0 = cellsOnEdge(iEdge,0) - 1
         cell1 = cellsOnEdge(iEdge,1) - 1

         if (boundaryCellMark(cell0) .EQ. 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = cell0
            valEntries(iEntry) = -1./dcEdge(iEdge)
            iEntry = iEntry + 1
         end if

         if (boundaryCellMark(cell1) .EQ. 0) then
            rows(iEntry) = iEdge
            cols(iEntry) = cell1
            valEntries(iEntry) = 1./dcEdge(iEdge)
            iEntry = iEntry + 1
         end if
      end do
   end if

   nEntries = iEntry

end subroutine construct_matrix_discrete_skewgrad_td


! Discrete skewgrad in the normal direction, with natural boundary condition
! This type of boundary condition is required to ensure the symmetry of the
! Hamiltonian system, and hence the conservations. 
subroutine discrete_skewgrad_nnat(nEdges, nVertices, nCells, &
     scalar_vertex, scalar_cell, verticesOnEdge, cellsOnEdge, dvEdge, &
     skewgrad_n)
  
  integer, intent(in) :: nEdges, nVertices, nCells
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1), cellsOnEdge(0:nEdges-1,0:1)
  real*8, intent(in)  :: dvEdge(0:nEdges-1)
  real*8, intent(in)  :: scalar_vertex(0:nVertices-1), scalar_cell(0:nCells-1)
  real*8, intent(out) :: skewgrad_n(0:nEdges-1)

  integer :: iEdge, vertex0, vertex1, cell0, cell1
  double precision :: scalar_edge

  skewgrad_n = 0.0
  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1
        if (vertex0 .GE. 0 .and. vertex1 .GE. 0) then
           skewgrad_n(iEdge) = (scalar_vertex(vertex0) - scalar_vertex(vertex1))/dvEdge(iEdge)
        else if (vertex0 .GE. 0) then
           cell0 = cellsOnEdge(iEdge,0) - 1
           cell1 = cellsOnEdge(iEdge,1) - 1
           scalar_edge = 0.5*(scalar_cell(cell0) + scalar_cell(cell1))
           skewgrad_n(iEdge) =  (scalar_vertex(vertex0) - scalar_edge) /dvEdge(iEdge)
           
        else if (vertex1 .GE. 0) then
           cell0 = cellsOnEdge(iEdge,0) - 1
           cell1 = cellsOnEdge(iEdge,1) - 1
           scalar_edge = 0.5*(scalar_cell(cell0) + scalar_cell(cell1))
           skewgrad_n(iEdge) =  (scalar_edge - scalar_vertex(vertex1)) /dvEdge(iEdge)

        else
           write(*,*) "Vertex indices in verticesOnEdge are wrong in discrete_skewgrad_nnat. Exit."
           stop
        end if
  end do

end subroutine discrete_skewgrad_nnat


!
! Functions and subroutines for the Shallow Water Test Case #8 (barotropic instability)
!

function swtc8_u(theta)
  double precision :: theta
  double precision, parameter :: umax = 80.
  double precision, parameter :: pi = 3.141592653589793
  double precision, parameter :: lat0 = pi/7
  double precision, parameter :: lat1 = pi/2 - lat0
  double precision, parameter :: en = exp(-4./(lat1-lat0)**2)
  
  double precision :: swtc8_u, u

  if (theta <= lat0) then
     u = 0.
  else if (theta >= lat1) then
     u = 0.
  else
     u = exp(1./(theta - lat0)/(theta - lat1))
     u = u * umax / en
  end if

  swtc8_u = u
  
end function swtc8_u


function swtc8_vort(theta)
  double precision :: theta
  double precision, parameter :: pi = 3.141592653589793
  double precision, parameter :: lat0 = pi/7
  double precision, parameter :: lat1 = pi/2 - lat0
  double precision, parameter :: a = 6371000.0

  double precision :: swtc8_vort, factor

  if (theta <= lat0) then
     swtc8_vort = 0.
  else if (theta >= lat1) then
     swtc8_vort = 0.
  else
     factor = (2*theta - lat0 -lat1)/(theta - lat0)**2/(theta-lat1)**2
     factor = factor + tan(theta)
     factor = factor
     swtc8_vort = swtc8_u(theta)/a
     swtc8_vort = swtc8_vort * factor
  end if

end function swtc8_vort

subroutine compute_swtc8_vort(nCells,   &
     latCell,   &
     vorticity)
  integer, intent(in)  :: nCells
  double precision, intent(in) :: latCell(0:nCells-1)
  double precision, intent(out) :: vorticity(0:nCells-1)

  integer :: iCell

  do iCell = 0, nCells-1
     vorticity(iCell) = swtc8_vort(latCell(iCell))
  end do

end subroutine compute_swtc8_vort

subroutine compute_swtc8_gh(nCells,  &
     latCell,  &
     gh)
  integer, intent(in) :: nCells
  double precision, intent(in) :: latCell(0:nCells-1)
  double precision, intent(out) :: gh(0:nCells-1)
  double precision, parameter :: pi = 3.141592653589793
  double precision, parameter :: lat0 = pi/7
  double precision, parameter :: lat1 = pi/2 - lat0
  double precision, parameter :: a = 6371000.0
  double precision, parameter :: Omega = 7.292e-5
  integer, parameter :: ngpts = 63

  double precision :: J, theta_end
  double precision, dimension(ngpts) :: gpts
  double precision, dimension(ngpts) :: ref_gpts
  double precision, dimension(ngpts) :: gwts

  integer :: iCell, i

!  if (ngpts == 13) then
!   gwts = &
!       (/ 0.2325515532308739, &
!       0.2262831802628972, &
!       0.2262831802628972, &
!       0.2078160475368885, &
!       0.2078160475368885, &
!       0.1781459807619457, &
!       0.1781459807619457, &
!       0.1388735102197872, &
!       0.1388735102197872, &
!       0.0921214998377285, &
!       0.0921214998377285, &
!       0.0404840047653159, &
!       0.0404840047653159/)
!  ref_gpts = &
!       (/0.0000000000000000, &
!       -0.2304583159551348, &
!       0.2304583159551348, &
!       -0.4484927510364469, &
!       0.4484927510364469, &
!       -0.6423493394403402, &
!       0.6423493394403402, &
!       -0.8015780907333099, &
!       0.8015780907333099, &
!       -0.9175983992229779, &
!       0.9175983992229779, &
!       -0.9841830547185881, &
!       0.9841830547185881/)
!else if (ngpts == 63) then
   gwts = (/ &
	0.0494723666239310, &
	0.0494118330399182, &
	0.0494118330399182, &
	0.0492303804237476, &
	0.0492303804237476, &
	0.0489284528205120, &
	0.0489284528205120, &
	0.0485067890978838, &
	0.0485067890978838, &
	0.0479664211379951, &
	0.0479664211379951, &
	0.0473086713122689, &
	0.0473086713122689, &
	0.0465351492453837, &
	0.0465351492453837, &
	0.0456477478762926, &
	0.0456477478762926, &
	0.0446486388259414, &
	0.0446486388259414, &
	0.0435402670830276, &
	0.0435402670830276, &
	0.0423253450208158, &
	0.0423253450208158, &
	0.0410068457596664, &
	0.0410068457596664, &
	0.0395879958915441, &
	0.0395879958915441, &
	0.0380722675843496, &
	0.0380722675843496, &
	0.0364633700854573, &
	0.0364633700854573, &
	0.0347652406453559, &
	0.0347652406453559, &
	0.0329820348837793, &
	0.0329820348837793, &
	0.0311181166222198, &
	0.0311181166222198, &
	0.0291780472082805, &
	0.0291780472082805, &
	0.0271665743590979, &
	0.0271665743590979, &
	0.0250886205533450, &
	0.0250886205533450, &
	0.0229492710048899, &
	0.0229492710048899, &
	0.0207537612580391, &
	0.0207537612580391, &
	0.0185074644601613, &
	0.0185074644601613, &
	0.0162158784103383, &
	0.0162158784103383, &
	0.0138846126161156, &
	0.0138846126161156, &
	0.0115193760768800, &
	0.0115193760768800, &
	0.0091259686763267, &
	0.0091259686763267, &
	0.0067102917659601, &
	0.0067102917659601, &
	0.0042785083468638, &
	0.0042785083468638, &
	0.0018398745955771, &
	0.0018398745955771 /)
   ref_gpts = (/ &
	0.0000000000000000, &
	-0.0494521871161596, &
	0.0494521871161596, &
	-0.0987833564469453, &
	0.0987833564469453, &
	-0.1478727863578720, &
	0.1478727863578720, &
	-0.1966003467915067, &
	0.1966003467915067, &
	-0.2448467932459534, &
	0.2448467932459534, &
	-0.2924940585862514, &
	0.2924940585862514, &
	-0.3394255419745844, &
	0.3394255419745844, &
	-0.3855263942122479, &
	0.3855263942122479, &
	-0.4306837987951116, &
	0.4306837987951116, &
	-0.4747872479948044, &
	0.4747872479948044, &
	-0.5177288132900333, &
	0.5177288132900333, &
	-0.5594034094862850, &
	0.5594034094862850, &
	-0.5997090518776252, &
	0.5997090518776252, &
	-0.6385471058213654, &
	0.6385471058213654, &
	-0.6758225281149861, &
	0.6758225281149861, &
	-0.7114440995848458, &
	0.7114440995848458, &
	-0.7453246483178474, &
	0.7453246483178474, &
	-0.7773812629903724, &
	0.7773812629903724, &
	-0.8075354957734567, &
	0.8075354957734567, &
	-0.8357135543195029, &
	0.8357135543195029, &
	-0.8618464823641238, &
	0.8618464823641238, &
	-0.8858703285078534, &
	0.8858703285078534, &
	-0.9077263027785316, &
	0.9077263027785316, &
	-0.9273609206218432, &
	0.9273609206218432, &
	-0.9447261340410098, &
	0.9447261340410098, &
	-0.9597794497589419, &
	0.9597794497589419, &
	-0.9724840346975701, &
	0.9724840346975701, &
	-0.9828088105937273, &
	0.9828088105937273, &
	-0.9907285468921895, &
	0.9907285468921895, &
	-0.9962240127779701, &
	0.9962240127779701, &
	-0.9992829840291237, &
	0.9992829840291237 /)
 ! end if

  do iCell = 0, nCells-1
     if (latCell(iCell) <= lat0) then
        gh(iCell) = -.0
     else
        theta_end = min(lat1, latCell(iCell))
        gpts = ref_gpts * (theta_end - lat0)/2 + (lat0 + theta_end)/2
        J = (theta_end - lat0)/2

        gh(iCell) = 0.
        do i = 1, ngpts
           gh(iCell) = gh(iCell) + gwts(i) * swtc8_u(gpts(i)) * (2*Omega*sin(gpts(i)) + &
                tan(gpts(i)) * swtc8_u(gpts(i)) / a)
        end do
        gh(iCell) = -a*J*gh(iCell)
     end if
  end do
  
end subroutine compute_swtc8_gh

end module
