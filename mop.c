
static char help[] = "A vorticity-divergence based Shallow Water Solver";

#include <petscksp.h>

/*
    User-defined context that contains all the data structures used
    in the linear solution process.
*/

typedef struct {
  /* Mesh dimensions */
  int         nCells;
  int         nEdges;
  int         nVertices;
  int         vertexDegree;
  int         nVertLevels;

  double      *xCell;
  double      *yCell;
  double      *zCell;
  double      *xEdge;
  double      *yEdge;
  double      *zEdge;
  double      *xVertex;
  double      *yVertex;
  double      *zVertex;
  double      *latCell;
  double      *lonCell;
  double      *latVertex;
  double      *lonVertex;

  double      *dcEdeg;
  double      *dvEdeg;
  double      *areaCell;
  double      *areaTriangle;
  double      **kiteAreaOnVertex;

  int         *nEdgesOnCell;
  int         *cellsOnEdge;
  int         **cellsOnCell;
  int         **verticesOnCell;
  int         **edgesOnCell;
  int         **verticesOnEdge;
  int         **edgesOnVertex;
  int         **cellsOnVertex;

  int         *boundaryEdgeMark;
  int         *boundaryCellMark;

  double      *fCell;
  double      *fEdge;
} Mesh;



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       m = 6,n = 7,t,tmax = 2,i,Ii,j,N;
  PetscScalar    *userx,*rho,*solution,*userb,hx,hy,x,y;
  PetscReal      enorm;
  /*
     Initialize the PETSc libraries
  */
  PetscInitialize(&argc,&args,(char*)0,help);

  return 0;
}
