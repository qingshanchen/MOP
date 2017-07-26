module swe_comp
implicit none

contains

subroutine compute_tend_pv_cell(nEdges, nCells, boundaryEdge, &
     cellsOnEdge, pv_edge, vorticity_cell, u, dcEdge, &
     dvEdge, areaCell, curlWind_cell, bottomDrag, &
     delVisc,  avgThickness, tend_pv_cell)

  implicit none

  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: boundaryEdge(0:nEdges-1)
  integer, intent(in) :: cellsOnEdge(0:nEdges-1,0:1)
  double precision, intent(in) :: pv_edge(0:nEdges-1), &
       dcEdge(0:nEdges-1), areaCell(0:nCells-1), &
       u(0:nEdges-1), avgThickness
  double precision, intent(in) :: dvEdge(0:nEdges-1),  &
       delVisc, curlWind_cell(0:nCells-1), &
       vorticity_cell(0:nCells-1), bottomDrag
  double precision, intent(out):: tend_pv_cell(0:nCells-1)

  double precision :: gradVorticityN(0:nEdges-1), &
       laplaceVORTICITY_cell(0:nCells-1), &
       gradLaplaceVorticityN(0:nEdges-1), &
       laplaceLaplaceVORTICITY_cell(0:nCells-1)
  
  integer :: iEdge, cell0, cell1, iCell

  double precision :: u_pv_edge(0:nEdges-1), pv_flux(0:nCells-1), del2Visc

  tend_pv_cell(:) = curlWind_cell(:) / avgThickness - bottomDrag * vorticity_cell(:)

  do iEdge = 0, nEdges-1
!     if (boundaryEdge(iEdge) .EQ. 0) then
         cell0 = cellsOnEdge(iEdge,0) - 1
         cell1 = cellsOnEdge(iEdge,1) - 1

         tend_pv_cell(cell0) = tend_pv_cell(cell0) - &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell0)
         tend_pv_cell(cell1) = tend_pv_cell(cell1) + &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell1)
!      endif
   end do

  ! Compute gradient of VORTICITY in the normal direction
  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        gradVorticityN(iEdge) = (vorticity_cell(cell1) - vorticity_cell(cell0))/dcEdge(iEdge)
     else
        gradVorticityN(iEdge) = 0.0 ! Zero flux on solid boundary
     end if
  end do

  ! Compute Laplace of VORTICITY on each cell
  laplaceVORTICITY_cell(:) = 0.0

  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        laplaceVORTICITY_cell(cell0) = laplaceVORTICITY_cell(cell0) + &
             gradVorticityN(iEdge) * dvEdge(iEdge)
        laplaceVORTICITY_cell(cell1) = laplaceVORTICITY_cell(cell1) - &
             gradVorticityN(iEdge) * dvEdge(iEdge)
     end if
  end do

  do iCell = 0, nCells-1
     laplaceVORTICITY_cell(iCell) = laplaceVORTICITY_cell(iCell)/areaCell(iCell)
  end do

  ! Compute gradient of Laplace VORTICITY on each edge
  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        gradLaplaceVorticityN(iEdge) = (laplaceVorticity_cell(cell1) - &
             laplaceVORTICITY_cell(cell0))/dcEdge(iEdge)
     else

        gradLaplaceVorticityN(iEdge) = 0.0 ! Zero flux on solid boundary
     end if
  end do
  
  ! Compute Laplace Laplace VORTICITY on each cell
  laplaceLaplaceVORTICITY_cell(:) = 0.0

  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        laplaceLaplaceVORTICITY_cell(cell0) = laplaceLaplaceVORTICITY_cell(cell0) + &
             gradLaplaceVorticityN(iEdge) * dvEdge(iEdge)
        laplaceLaplaceVORTICITY_cell(cell1) = laplaceLaplaceVORTICITY_cell(cell1) - &
             gradLaplaceVorticityN(iEdge) * dvEdge(iEdge)
     end if
  end do

  do iCell = 0, nCells-1
     laplaceLaplaceVORTICITY_cell(iCell) = &
          laplaceLaplaceVORTICITY_cell(icell) / areaCell(iCell)
  end do

!  write(*,*) "delVisc = ", delVisc
!  write(*,*) "del2Visc = ", del2Visc
!  write(*,*) "min laplaceVORTICITY_cell = ", minval(laplaceVORTICITY_cell)
!  write(*,*) "max laplaceVORTICITY_cell = ", maxval(laplaceVORTICITY_cell)

  do iCell = 0, nCells-1
     tend_pv_cell(iCell) = tend_pv_cell(iCell) + &
          delVisc * laplaceVORTICITY_cell(iCell)
!     tend_pv_cell(iCell) = tend_pv_cell(iCell) - &
!          del2Visc * laplaceLaplaceVORTICITY_cell(iCell)
  end do

!  write(*,*) "tend_pv_cell(1152) = ", tend_pv_cell(1152)

end subroutine compute_tend_pv_cell


subroutine compute_pv_fluxes_cell(nEdges, nCells, &
     cellsOnEdge, pv_edge, u, dvEdge, areaCell, &
     pv_flux_cell)

  implicit none

  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1,0:1)
  double precision, intent(in) :: pv_edge(0:nEdges-1), &
       areaCell(0:nCells-1), &
       u(0:nEdges-1)
  double precision, intent(in) :: dvEdge(0:nEdges-1)
  double precision, intent(out):: pv_flux_cell(0:nCells-1)

  integer :: iEdge, cell0, cell1

  pv_flux_cell(:) = 0.0
  do iEdge = 0, nEdges-1
         cell0 = cellsOnEdge(iEdge,0) - 1
         cell1 = cellsOnEdge(iEdge,1) - 1

         pv_flux_cell(cell0) = pv_flux_cell(cell0) + &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell0)
         pv_flux_cell(cell1) = pv_flux_cell(cell1) - &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell1)
   end do

end subroutine compute_pv_fluxes_cell


subroutine compute_tend_eta_cell(nEdges, nCells, &
     boundaryEdge, cellsOnEdge, eta_edge, vorticity_cell, u, dcEdge, &
     dvEdge, areaCell, fEdge, bottomTopEdge, curlWind_cell, bottomDrag, &
     delVisc, btc, avgThickness, &
     tend_eta_cell)

  implicit none

  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: boundaryEdge(0:nEdges-1)
  integer, intent(in) :: cellsOnEdge(0:nEdges-1,0:1)
  double precision, intent(in) :: eta_edge(0:nEdges-1), &
       dcEdge(0:nEdges-1), areaCell(0:nCells-1), &
       u(0:nEdges-1), avgThickness, fEdge(0:nEdges-1), &
       bottomTopEdge(0:nEdges-1)
  double precision, intent(in) :: dvEdge(0:nEdges-1), btc, &
       delVisc, curlWind_cell(0:nCells-1), &
       vorticity_cell(0:nCells-1), bottomDrag
  double precision, intent(out):: tend_eta_cell(0:nCells-1)

  double precision :: gradVorticityN(0:nEdges-1), &
       laplaceVORTICITY_cell(0:nCells-1), &
       gradLaplaceVorticityN(0:nEdges-1), &
       laplaceLaplaceVORTICITY_cell(0:nCells-1)
  
  integer :: iEdge, cell0, cell1, iCell

  double precision :: pv_edge(0:nEdges-1)

  pv_edge(:) = eta_edge(:) + fEdge(:) + btc*bottomTopEdge(:)

  tend_eta_cell(:) = curlWind_cell(:) / avgThickness - bottomDrag * vorticity_cell(:)

  do iEdge = 0, nEdges-1
         cell0 = cellsOnEdge(iEdge,0) - 1
         cell1 = cellsOnEdge(iEdge,1) - 1

         tend_eta_cell(cell0) = tend_eta_cell(cell0) - &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell0)
         tend_eta_cell(cell1) = tend_eta_cell(cell1) + &
              pv_edge(iEdge)*u(iEdge)*dvEdge(iEdge)/areaCell(cell1)
   end do

  ! Compute gradient of VORTICITY in the normal direction
  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        gradVorticityN(iEdge) = (vorticity_cell(cell1) - vorticity_cell(cell0))/dcEdge(iEdge)
     else
        gradVorticityN(iEdge) = 0.0 ! Zero flux on solid boundary
     end if
  end do

  ! Compute Laplace of VORTICITY on each cell
  laplaceVORTICITY_cell(:) = 0.0

  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        laplaceVORTICITY_cell(cell0) = laplaceVORTICITY_cell(cell0) + &
             gradVorticityN(iEdge) * dvEdge(iEdge)
        laplaceVORTICITY_cell(cell1) = laplaceVORTICITY_cell(cell1) - &
             gradVorticityN(iEdge) * dvEdge(iEdge)
     end if
  end do

  do iCell = 0, nCells-1
     laplaceVORTICITY_cell(iCell) = laplaceVORTICITY_cell(iCell)/areaCell(iCell)
  end do

  do iCell = 0, nCells-1
     tend_eta_cell(iCell) = tend_eta_cell(iCell) + &
          delVisc * laplaceVORTICITY_cell(iCell)
  end do

end subroutine compute_tend_eta_cell


subroutine cell2vertex(nVertices, nCells, nEdges, vertexDegree, &
     cellsOnVertex, kiteAreasOnVertex, areaTriangle, verticesOnEdge, &
     scalar_cell, scalar_vertex)
  integer, intent(in) :: nVertices, vertexDegree, nCells, nEdges
  integer, intent(in) :: cellsOnVertex(0:nVertices-1, 0:vertexDegree-1), &
       &                 verticesOnEdge(0:nEdges-1,0:1)
  double precision, intent(in)  :: kiteAreasOnVertex(0:nVertices-1, 0:vertexDegree-1), &
       areaTriangle(0:nVertices-1), scalar_cell(0:nCells-1)
  double precision, intent(out) :: scalar_vertex(0:nVertices-1)

  integer :: iVertex, iCell, i, iEdge, vertex0, vertex1

  scalar_vertex(:) = 0.0
  do iVertex = 0, nVertices-1
      do i = 0, vertexDegree-1
         iCell = cellsOnVertex(iVertex, i) - 1
         scalar_vertex(iVertex) = scalar_vertex(iVertex) + &
              kiteAreasOnVertex(iVertex, i)*scalar_cell(iCell)/areaTriangle(iVertex)
      end do
  end do

end subroutine cell2vertex


! Compute the normal velocity component from psi_vertex and phi_cell
subroutine compute_u(nEdges, nVertices, nCells, verticesOnEdge, cellsOnEdge, dcEdge, dvEdge, phi_cell, psi_vertex, u)
  integer, intent(in) :: nEdges, nVertices, nCells
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1), cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1), dvEdge(0:nEdges-1), psi_vertex(0:nVertices-1), phi_cell(0:nCells-1)
  double precision, intent(out) :: u(0:nEdges-1)

  integer :: iEdge, vertex0, vertex1, cell0, cell1

  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        
        u(iEdge) = (phi_cell(cell1) - phi_cell(cell0)) / dcEdge(iEdge)
        
        if (vertex0 >= 0 .AND. vertex1 >= 0) then
           ! For interior edges
           u(iEdge) = u(iEdge) - (psi_vertex(vertex1) - psi_vertex(vertex0)) / dvEdge(iEdge)           

        else if (vertex0 < 0 .AND. vertex1 >= 0) then
           ! For partial edges on the boundary
        
           ! No-slip boundary condition, suitable when lateral diffusion is included.
           !u(iEdge) = 0.

           ! Constant value of zero for the stream function on the boundary
           u(iEdge) = u(iEdge) - (psi_vertex(vertex1) - 0. )/dvEdge(iEdge)
           
        else if (vertex0 >= 0 .AND. vertex1 < 0) then
           ! For partial edges on the boundary

           ! No-slip boundary condition, suitable when lateral diffusion is included.
           !u(iEdge) = 0.

           ! Constant value of zero for the stream function on the boundary
           u(iEdge) = u(iEdge) - (0. - psi_vertex(vertex0)) / dvEdge(iEdge)
           
        else
           ! Error
           write(*,*) "Error in compute_u. Exit"
           stop
           
        end if
  end do

end subroutine


! Compute the tangential velocity component v from psi_cell and phi_vertex
subroutine compute_v(nEdges, nVertices, nCells, verticesOnEdge, cellsOnEdge, dcEdge, dvEdge, phi_vertex, psi_cell, v)
  integer, intent(in) :: nEdges, nVertices, nCells
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1), cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1), dvEdge(0:nEdges-1), psi_cell(0:nVertices-1), phi_vertex(0:nCells-1)
  double precision, intent(out) :: v(0:nEdges-1)

  integer :: iEdge, vertex0, vertex1, cell0, cell1

  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        
        v(iEdge) = (psi_cell(cell1) - psi_cell(cell0)) / dcEdge(iEdge)
        
        if (vertex0 >= 0 .AND. vertex1 >= 0) then
           ! For interior edges
           v(iEdge) = v(iEdge) + (phi_vertex(vertex1) - phi_vertex(vertex0)) / dvEdge(iEdge)           

        else if (vertex0 < 0 .AND. vertex1 >= 0) then
           ! For partial edges on the boundary, homogeneous Neumann for phi
           v(iEdge) = v(iEdge) + 0.
           
        else if (vertex0 >= 0 .AND. vertex1 < 0) then
           ! For partial edges on the boundary, homogeneous Neumann for phi
           v(iEdge) = v(iEdge) + 0.
           
        else
           ! Error
           write(*,*) "Error in compute_v. Exit"
           stop
           
        end if
  end do

end subroutine compute_v


! Compute scalar_edge from scalar_cell (for now)
subroutine cell2edge(nEdges, nCells, cellsOnEdge, cellBoundaryMark, scalar_cell, scalar_edge)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1), cellBoundaryMark(0:nCells-1)
  real*8, intent(in)  :: scalar_cell(0:nCells-1)
  real*8, intent(out) :: scalar_edge(0:nEdges-1)

  integer :: iEdge, iCell1, iCell2

  do iEdge = 0, nEdges-1
        iCell1 = cellsOnEdge(iEdge,0) - 1
        iCell2 = cellsOnEdge(iEdge,1) - 1

        scalar_edge(iEdge) = 0.5*(scalar_cell(iCell1) + scalar_cell(iCell2))
  end do

end subroutine


! Compute pv_edge from pv_cell (for now)
subroutine compute_pv_edge_apvm(nEdges, nCells, cellsOnEdge, cellBoundaryMark, dcEdge, dt, pv_cell, u, apvm_factor, pv_edge)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1), cellBoundaryMark(0:nCells-1)
  real*8, intent(in)  :: pv_cell(0:nCells-1), u(0:nEdges-1), dcEdge(0:nEdges-1)
  double precision, intent(in) :: apvm_factor, dt
  real*8, intent(out) :: pv_edge(0:nEdges-1)

  integer :: iEdge, iCell1, iCell2
  double precision    :: grad_pv(0:nEdges-1)

  call discrete_grad_n(nEdges, nCells, pv_cell, cellsOnEdge, dcEdge, grad_pv)

  do iEdge = 0, nEdges-1
        iCell1 = cellsOnEdge(iEdge,0) - 1
        iCell2 = cellsOnEdge(iEdge,1) - 1

        pv_edge(iEdge) = 0.5*(pv_cell(iCell1) + pv_cell(iCell2))
        pv_edge(iEdge) = pv_edge(iEdge) - apvm_factor * dt * u(iEdge) * grad_pv(iEdge)
  end do

end subroutine


! Compute pv_edge from pv_cell (for now)
subroutine compute_pv_edge_upwind(nEdges, nCells, cellsOnEdge, pv_cell, u, pv_edge)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: pv_cell(0:nCells-1), u(0:nEdges-1)
  real*8, intent(out) :: pv_edge(0:nEdges-1)

  integer :: iEdge, iCell1, iCell2

  do iEdge = 0, nEdges-1
        iCell1 = cellsOnEdge(iEdge,0) - 1
        iCell2 = cellsOnEdge(iEdge,1) - 1

        if (u(iEdge) > 0) then
           pv_edge(iEdge) = pv_cell(iCell1)
        else
           pv_edge(iEdge) = pv_cell(iCell2)
        end if
  end do

end subroutine


subroutine discrete_grad_n(nEdges, nCells, scalar_cell, cellsOnEdge, &
     dcEdge, grad_n)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: dcEdge(0:nEdges-1)
  real*8, intent(in)  :: scalar_cell(0:nCells-1)
  real*8, intent(out) :: grad_n(0:nEdges-1)

  integer :: iEdge, cell0, cell1

  grad_n = 0.0
  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1
        grad_n(iEdge) = (scalar_cell(cell1) - scalar_cell(cell0))/dcEdge(iEdge)
  end do

end subroutine discrete_grad_n


subroutine discrete_grad_t(nEdges, nVertices, &
     scalar_vertex, verticesOnEdge, dvEdge, &
     grad_t)
  
  integer, intent(in) :: nEdges, nVertices
  integer, intent(in) :: verticesOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: dvEdge(0:nEdges-1)
  real*8, intent(in)  :: scalar_vertex(0:nVertices-1)
  real*8, intent(out) :: grad_t(0:nEdges-1)

  integer :: iEdge, vertex0, vertex1

  grad_t = 0.0
  do iEdge = 0, nEdges-1
        vertex0 = verticesOnEdge(iEdge,0) - 1
        vertex1 = verticesOnEdge(iEdge,1) - 1
        if (vertex0 .GE. 0 .and. vertex1 .GE. 0) then
           grad_t(iEdge) = (scalar_vertex(vertex1) - scalar_vertex(vertex0))/dvEdge(iEdge)
        else if (vertex0 .GE. 0) then
           grad_t(iEdge) =  - scalar_vertex(vertex0)/dvEdge(iEdge)
        else if (vertex1 .GE. 0) then
           grad_t(iEdge) =  scalar_vertex(vertex1)/dvEdge(iEdge)
        else
           write(*,*) "Vertex indices in verticesOnEdge are wrong in discrete_grad_t. Exit."
           stop
        end if
  end do

end subroutine discrete_grad_t

! Given a discrete vector field vector_n, compute its discrete divergence.
! The orientation on the edge is assumed to be from cell0 (first cell) to cell1 (second cell)
subroutine discrete_div(nEdges, nCells, vector_n, cellsOnEdge, &
     dvEdgeInterior, areaCell, divergence_n)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: vector_n(0:nEdges-1), dvEdgeInterior(0:nEdges-1), &
       areaCell(0:nCells-1)
  real*8, intent(out) :: divergence_n(0:nCells-1)

  integer :: iEdge, cell0, cell1, iCell

  divergence_n(:) = 0.0

  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        divergence_n(cell0) = divergence_n(cell0) + vector_n(iEdge) * dvEdgeInterior(iEdge)
        divergence_n(cell1) = divergence_n(cell1) - vector_n(iEdge) * dvEdgeInterior(iEdge)
  end do

  do iCell = 0, nCells-1
     divergence_n(iCell) = divergence_n(iCell)/areaCell(iCell)
  end do
  
end subroutine discrete_div


! Given a discrete vector field vector_t, compute its discrete divergence.
! The orientation on the edge is such that the first cell (cell0) appears on the left of the edge
subroutine discrete_curl(nEdges, nCells, vector_t, cellsOnEdge, &
     dvEdge, areaCell, curl)
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  real*8, intent(in)  :: vector_t(0:nEdges-1), dvEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  real*8, intent(out) :: curl(0:nCells-1)

  integer :: iEdge, cell0, cell1, iCell

  curl(:) = 0.0

  do iEdge = 0, nEdges-1
        cell0 = cellsOnEdge(iEdge,0) - 1
        cell1 = cellsOnEdge(iEdge,1) - 1

        curl(cell0) = curl(cell0) + vector_t(iEdge) * dvEdge(iEdge)
        curl(cell1) = curl(cell1) - vector_t(iEdge) * dvEdge(iEdge)

  end do

  do iCell = 0, nCells-1
     curl(iCell) = curl(iCell)/areaCell(iCell)
  end do
  
end subroutine discrete_curl



subroutine discrete_laplace_cell(nEdges, nCells,  &
     cellsOnEdge, dcEdge, dvEdge, areaCell, scalar_cell, &
          laplace_cell)
  
  integer, intent(in) :: nEdges, nCells
  integer, intent(in) :: cellsOnEdge(0:nEdges-1, 0:1)
  double precision, intent(in)  :: dcEdge(0:nEdges-1), dvEdge(0:nEdges-1), &
       areaCell(0:nCells-1), scalar_cell(0:nCells-1)
  double precision, intent(out) :: laplace_cell(0:nCells-1)

  double precision              :: grad_n(0:nEdges-1)

  call discrete_grad_n(nEdges, nCells, &
     scalar_cell, cellsOnEdge, dcEdge, &
     grad_n)

   call discrete_div(nEdges, nCells, &
                       grad_n, cellsOnEdge, dvEdge, areaCell, &
                       laplace_cell)
end subroutine discrete_laplace_cell
  

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


subroutine construct_discrete_laplace_interior(nCells, nEdges, &
     boundaryEdge, cellsOnEdge, boundaryCell, cellRankInterior, &
     dvEdge, dcEdge, areaCell, nEntries, rows, &
     cols, valEntries)

  integer, intent(in)    :: nCells, nEdges
  integer, intent(in)    :: boundaryEdge(0:nEdges-1), &
       cellsOnEdge(0:nEdges-1,0:1), boundaryCell(0:nCells-1), &
       cellRankInterior(0:nCells-1)
  real*8, intent(in)     :: dvEdge(0:nEdges-1), dcEdge(0:nEdges-1), &
       areaCell(0:nCells-1)
  integer, intent(out)   :: rows(0:4*nEdges+nCells-1), cols(0:4*nEdges+nCells-1), &
       nEntries
  real*8, intent(out)    :: valEntries(0:4*nEdges+nCells-1)

  integer   :: iEdge, iCell1, iCell2, iEntry, iCellInterior

  iEntry = 0

  do iEdge = 0, nEdges-1
     if (boundaryEdge(iEdge) .EQ. 0) then
        iCell1 = cellsOnEdge(iEdge,0) - 1
        iCell2 = cellsOnEdge(iEdge,1) - 1

        if (boundaryCell(iCell1) .EQ. 0) then
           iCellInterior = cellRankInterior(iCell1) - 1
           rows(iEntry) = iCellInterior
           cols(iEntry) = iCell1
           valEntries(iEntry) = -dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell1)
           iEntry = iEntry + 1

           rows(iEntry) = iCellInterior
           cols(iEntry) = iCell2
           valEntries(iEntry) = dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell1)
           iEntry = iEntry + 1
        end if
        

        if (boundaryCell(iCell2) .EQ. 0) then
           iCellInterior = cellRankInterior(iCell2) - 1              
           rows(iEntry) = iCellInterior
           cols(iEntry) = iCell1
           valEntries(iEntry) = dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell2)
           iEntry = iEntry + 1

           rows(iEntry) = iCellInterior
           cols(iEntry) = iCell2
           valEntries(iEntry) = -dvEdge(iEdge)/dcEdge(iEdge)/areaCell(iCell2)
           iEntry = iEntry + 1
        end if
     end if
  end do

  nEntries = iEntry
  
end subroutine construct_discrete_laplace_interior


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
        valEntries(iEntry) = 0.
     end if
  end do
  
end subroutine construct_discrete_laplace_neumann


end module
