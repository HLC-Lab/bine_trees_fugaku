#include <mpi.h>
#include <stdio.h>
#include "../lib/fugaku/bine_utofu.h"

#define NUM_BYTES_PER_SEND 128
#define BUFFER_NELEM 4194304

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  printf("MPI Initialized\n");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double starttime, endtime_tofu, endtime_mpi;

  // Ping 
  char* sbuffer = (char*) malloc(sizeof(char)*BUFFER_NELEM);
  char* rbuffer = (char*) malloc(sizeof(char)*BUFFER_NELEM);
  char* vbuffer = (char*) malloc(sizeof(char)*BUFFER_NELEM);  

  srand(time(NULL) + rank);

  for(size_t i = 0; i < BUFFER_NELEM; i++){
    sbuffer[i] = rand();
  }
  
  if(rank == 0){
    bine_utofu_comm_descriptor* desc = bine_utofu_setup_communication(0, 1, sbuffer, sizeof(char)*BUFFER_NELEM, rbuffer, sizeof(char)*BUFFER_NELEM);
    starttime = MPI_Wtime();      
    bine_utofu_isend(desc);
    bine_utofu_wait(desc);
    endtime_tofu = MPI_Wtime();      
    MPI_Sendrecv(sbuffer, BUFFER_NELEM, MPI_CHAR, 1, 0, vbuffer, BUFFER_NELEM, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
    endtime_mpi = MPI_Wtime();      
    bine_utofu_destroy_communication(desc); 
  }else if(rank == 1){
    bine_utofu_comm_descriptor* desc = bine_utofu_setup_communication(0, 0, sbuffer, sizeof(char)*BUFFER_NELEM, rbuffer, sizeof(char)*BUFFER_NELEM);
    starttime = MPI_Wtime();      
    bine_utofu_isend(desc);
    bine_utofu_wait(desc);
    endtime_tofu = MPI_Wtime();          
    MPI_Sendrecv(sbuffer, BUFFER_NELEM, MPI_CHAR, 0, 0, vbuffer, BUFFER_NELEM, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    endtime_mpi = MPI_Wtime();      
    bine_utofu_destroy_communication(desc); 
  }

  for(size_t i = 0; i < BUFFER_NELEM; i++){
    assert(rbuffer[i] == vbuffer[i]);
  }
  
  free(sbuffer);
  free(rbuffer);
  free(vbuffer);

  printf("Tofu time (usec): %lf MPI Time (usec): %lf \n", (endtime_tofu - starttime)*1000000.0, (endtime_mpi - endtime_tofu)*1000000.0);
  MPI_Finalize();
  return 0;
}
