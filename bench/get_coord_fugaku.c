#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h>              // Include header file
#define LDIM 3
#define TDIM 6

int main(int argc, char *argv[])
{
  int size, myrank, i, j, x, y, z;
  int mydimension;
  int coords[LDIM], tcoords[TDIM];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  FJMPI_Topology_get_dimension(&mydimension);
  FJMPI_Topology_get_shape(&x, &y, &z);

  if (myrank == 0) {
     printf("My Dimension= %d\n",mydimension);
     printf("My Shape: X= %d", x);
     if (y != 0) printf(", Y= %d", y);
     if (z != 0) printf(", Z= %d", z);
     printf("\n\n");
     for ( i=0; i < size ; i++){
       FJMPI_Topology_get_coords(MPI_COMM_WORLD, i, FJMPI_LOGICAL, mydimension, coords);
       FJMPI_Topology_get_coords(MPI_COMM_WORLD, i, FJMPI_TOFU_SYS, TDIM, tcoords);
       switch(mydimension) {
         case 1:
                printf("rank to x : rank= %d, (X)=( %d ) ",i, coords[0]);
                break;
         case 2:
                printf("rank to xy : rank= %d, (X,Y)=( %d, %d ) ",i, coords[0], coords[1]);
                break;
         case 3:
                printf("rank to xyz : rank= %d, (X,Y,Z)=( %d, %d, %d ) ", i, coords[0], coords[1], coords[2]);
                break;
         default:
                break;
        }
       printf("(x,y,z,a,b,c)=(");
       for ( j=0; j < TDIM-1; j++) {
              printf("%d,", tcoords[j]);
       }
       printf("%d)\n",tcoords[TDIM-1]);
     }
  }

  MPI_Finalize();
  return 0;
}
