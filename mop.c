
static char help[] = "A vorticity-divergence based Shallow Water Solver";

#include <petscksp.h>
#include <netcdf.h>

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

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


typedef struct {
    char      *grid_file_name;    

  double    dt;
  
} Parameter;

int moc_initialize_parameters(Parameter *cp ){
    
    cp->grid_file_name = "grid.nc";
    cp->dt = 172.8;

    return 0;
}
      

int read_grid(Parameter *cptr, Mesh *gptr){
  int          ncid, dimid, varid, retval;

  if ((retval = nc_open(cptr->grid_file_name, NC_NOWRITE, &ncid)))
    ERR(retval);

  if ((retval = nc_close(ncid)))
    ERR(retval);

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       m = 6,n = 7,t,tmax = 2,i,Ii,j,N;
  PetscScalar    *userx,*rho,*solution,*userb,hx,hy,x,y;
  PetscReal      enorm;
  Parameter      c;
  Mesh           g;
  
  /*
     Initialize the PETSc libraries
  */
  PetscInitialize(&argc,&args,(char*)0,help);

  moc_initialize_parameters(&c);
  
  ierr = read_grid(&c, &g);
  
  return 0;
}
