#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

double *alloc_1d_double(int n){
  double *p;

  p = (double *) malloc(n*sizeof(double));

  if (p == NULL){
    fprintf(stderr, "could not allocate memory.");
    exit(1);
  }

  return p;
}

void free_1d_double(double *p){
  free(p);
}

void free_2d_double(double **p){
  free(p[0]);
  free(p);
}

void free_2d_int(int **p){
  free(p[0]);
  free(p);
}

double **alloc_2d_double(int m, int n){
  double **p;
  int i;

  p = (double **) malloc((size_t) m*sizeof(double *));
  if (!p){
    fprintf(stderr, "could not allocate memory for matrix %d by %d", m, n);
    exit(1);
  }

  p[0] = (double *) malloc((size_t) m*n*sizeof(double));
  if (!p[0]){
    fprintf(stderr, "could not allocate memory for matrix %d by %d", m, n);
    exit(1);
  }

  for (i=1; i<m; i++)
    p[i] = p[i-1] + n;

  return p;
}

int **alloc_2d_int(int m, int n){
  int **p;
  int i;

  p = (int **) malloc((size_t) m*sizeof(int *));
  if (!p){
    fprintf(stderr, "could not allocate memory for integer matrix %d by %d", m, n);
    exit(1);
  }

  p[0] = (int *) malloc((size_t) m*n*sizeof(int));
  if (!p[0]){
    fprintf(stderr, "could not allocate memory for integer matrix %d by %d", m, n);
    exit(1);
  }

  for (i=1; i<m; i++)
    p[i] = p[i-1] + n;

  return p;
}

