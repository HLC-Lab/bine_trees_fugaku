/***************************************************************************
 *   Name Seydou BA
 ***************************************************************************/

/***************************************************************************
AllreduceRSAG implementation from:
https://github.com/andrea-garritano/allreduce/blob/master/allreduce.cpp

RingAllreduce from mpich and baidu github:
https://github.com/pmodels/mpich/blob/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce
https://github.com/baidu-research/baidu-allreduce/blob/master/collectives.cu

Bucket allreduce based on the one used for swing-allreduce-sim:
https://github.com/HLC-Lab/swing-allreduce-sim/blob/98ad4ebd58b54d4de55291d9d8fc1731a4a49bd3/sst-elements-library-11.1.0/src/sst/elements/ember/mpi/motifs/emberhxmesh.cc#L975

 ***************************************************************************/

#include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <sstream>

#include<vector>
#include <cassert>

using namespace std;

// Allocate a new memory buffer on CPU or GPU.
double* alloc(size_t size) {
    return new double[size];
}
// Deallocate an allocated memory buffer.
void dealloc(double* buffer) {
    delete[] buffer;
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(double* dst, double* src, size_t size) {
    std::memcpy((void*) dst, (void*) src, size * sizeof(double));
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(double* dst, double* src, size_t size) {
    // Accumulate values from `src` into `dst` on the CPU.
    for(size_t i = 0; i < size; i++) {
        dst[i] += src[i];
    }
}

void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    if(datatype == MPI_INT32_T){
        const int32_t *in = (const int32_t *)inbuf;
        int32_t *inout = (int32_t *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            //fprintf(stderr, "Unknown reduction op\n");
            //exit(EXIT_FAILURE);
            // We assume this is the custom VOID operator
        }
    }else if(datatype == MPI_INT){
        const int *in = (const int *)inbuf;
        int *inout = (int *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_CHAR){
        const char *in = (const char *)inbuf;
        char *inout = (char *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_FLOAT){
        const float *in = (const float *)inbuf;
        float *inout = (float *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_DOUBLE){
        const double *in = (const double *)inbuf;
        double *inout = (double *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else{
        fprintf(stderr, "Unknown reduction datatype\n");
        exit(EXIT_FAILURE);
    }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
    std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}

/**
 * Wrapper function around MPI_Allreduce (leave this unchanged)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduce(double *sendbuf, double *recvbuf, int count)
{
	MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * Wrapper function around MPI_Allreduce (implement reduce-scatter / allgather algorithm)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduceRSAG(double *sendbuf, double *recvbuf, int count)
{
	// Note: assume MPI_DOUBLE, MPI_SUM and MPI_COMM_WORLD parameters
	// These may be hard-coded in the implementation. You may assume
	// that the number of processes is a power of two and you may assume
	// that "count" is a multiple of P.

	// Implement the allreduce algorithm using only point-to-point
	// communications, using the reduce-scatter / allgather algorithm
	// (see exercises slides for communication scheme)

	// Do not change the function signature as your implementation
	// will be automatically validated using a test framework.

	// Do not overwrite the sendbuf contents. Allocate a temp buffer
	// if necessary.

	int thisProc, nProc;
    MPI_Comm_rank( MPI_COMM_WORLD, &thisProc );
    MPI_Comm_size( MPI_COMM_WORLD, &nProc );
    MPI_Status status;

    int dest;
    int width;
    double *myData = new double[count];
    for (int i=0; i<count; i++)
        myData[i]=sendbuf[i];

    double *temp_sendbuf = new double[count];
    for (int i=0; i<count; i++)
        temp_sendbuf[i]=sendbuf[i];

    int nPhases = log2(nProc);

    for (int p = 0; p<nPhases; p++){
        width = (count/pow(2, p+1));

        if (thisProc%((int)pow(2, p+1))<(int)pow(2, p+1)/2){
            dest = thisProc + pow(2,p);
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i+width]; //second half of the message
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &status);

            for (size_t i = 0; i < width; i++)
                myData[i] += recvbuf[i]; //sum the replay
        }   
        else{
            dest = thisProc - pow(2,p);

            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &status);

            
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i]; //first half of the message

            for (size_t i = 0; i < width; i++)
                myData[i+width] += recvbuf[i]; //sum the replay
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            //Copy the second half on the first half
            for (size_t i = 0; i < width; i++)
                myData[i]=myData[i+width];
        }
    }
    for (int p = nPhases-1; p>=0; p--){
        width = (count/pow(2, p+1));
        if (thisProc%((int)pow(2, p+1))<(int)pow(2, p+1)/2){
            dest = thisProc + pow(2,p);
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i];
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &status);

            for (size_t i = 0; i < width; i++)
                myData[i+width] = recvbuf[i];
        }   
        else{
            dest = thisProc - pow(2,p);

            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &status);
            
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i];

            
            for (size_t i = 0; i < width; i++)
                myData[i+width] = myData[i];

            for (size_t i = 0; i < width; i++)
                myData[i] = recvbuf[i];
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

        }
    }

    for (size_t i = 0; i < count; i++)
        recvbuf[i] = myData[i];

    delete [] myData;
    delete [] temp_sendbuf;

}


/* Routine to schedule a ring exchange based allreduce. The algorithm is
 * based on Baidu's ring based allreduce. http://andrew.gibiansky.com/ */
//void RingAllreduce(float* data, size_t length, float** output_ptr)
//void allreduceRing(double *sendbuf, double *recvbuf, int count)
void allreduceRing(double* data, double* output, int count)
{
    // Get MPI size and rank.
    int rank, nProc;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nProc );
    
    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const int segment_size = count / nProc;
    std::vector<size_t> segment_sizes(nProc, segment_size);

    const size_t residual = count % nProc;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(nProc);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[nProc - 1] == count);

    // Allocate the output buffer.
 //   double* output = new double[count]; // alloc(count);
 //   output_ptr =  output;

    // Copy your data to the output buffer to avoid modifying the input buffer.
   // copy(output, data, count);
    std::memcpy((void*) output, (void*) data, count * sizeof(double));

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    double* buffer = alloc(segment_sizes[0]);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + nProc) % nProc;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % nProc;

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_DOUBLE;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).
    for (int i = 0; i < nProc - 1; i++) {
        int recv_chunk = (rank - i - 1 + nProc) % nProc;
        int send_chunk = (rank - i + nProc) % nProc;
        double* segment_send = &(output[segment_ends[send_chunk] -
                                   segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk],
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);

        double *segment_update = &(output[segment_ends[recv_chunk] -
                                         segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

        reduce(segment_update, buffer, segment_sizes[recv_chunk]);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(nProc - 1); ++i) {
        int send_chunk = (rank - i + 1 + nProc) % nProc;
        int recv_chunk = (rank - i + nProc) % nProc;
        // Segment to send - at every iteration we send segment (r+1-i)
        double* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        double* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }

    // Free temporary memory.
    dealloc(buffer);
}

#include <stdio.h>
#include <mpi-ext.h>              // Include header file
#define LDIM 3
#define TDIM 6

/***
// Convert a rank id into a list of d-dimensional coordinates
static void getCoordFromId(int id, int* dimensions, int dimensions_num, int* coord, int m_p){
    int nnodes = m_p;
    for (int i = 0; i < dimensions_num; i++) {
        nnodes = nnodes / dimensions[i];
        coord[i] = id / nnodes;
        id = id % nnodes;
    }
}

// Convert d-dimensional coordinates into a rank id
static int getIdFromCoord(int* coords, int* dimensions, int dimensions_num){
    int rank = 0;
    int multiplier = 1;
    int coord;
    for (int i = dimensions_num - 1; i >= 0; i--) {
        coord = coords[i];
        if (1) {            // cart_ptr->topo.cart.periodic[i]
            if (coord >= dimensions[i])
                coord = coord % dimensions[i];
            else if (coord < 0) {
                coord = coord % dimensions[i];
                if (coord)
                    coord = dimensions[i] + coord;
            }
        }
        rank += multiplier * coord;
        multiplier *= dimensions[i];
    }
    return rank;
}

static int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}
***/

void ringRedScatAG(double* data, int count, int nProc, int rank, int recvfrom, int sendto, int redscat){
    // Perform ring reduce-scatter or allgather on each line of a selected dimension.
    
    const int segment_size = count / nProc;
    std::vector<size_t> segment_sizes(nProc, segment_size);

    const size_t residual = count % nProc;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }
    
    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(nProc);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }
    // The last segment should end at the very end of the buffer.
    assert(segment_ends[nProc - 1] == count);
    
    // Allocate the output buffer.
    // The updated data set is used on subsequent reduce-scatter/allgather. 
     double* output = data;
  
    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    double* buffer = alloc(segment_sizes[0]);

    // Recv_from/send_to on each dimention is defined for each node before hand.
    const size_t recv_from = recvfrom;      // (rank - 1 + nProc) % nProc;  
    const size_t send_to = sendto;          //(rank + 1) % nProc;
    
    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_DOUBLE;
    int send_chunk, recv_chunk;

    // Selecting algorithm (0-1 reduce-scatter, 2-3 allgather), and direction (0,2 left dataset, 1,3 right dataset
    if(redscat<2){
        for (int i = 0; i < nProc - 1; i++) {
            if(redscat == 0){
                recv_chunk = (rank - i - 2 + nProc) % nProc;            // Where to receive data, rank = relcoord[current_d]
                send_chunk = (rank - i - 1 + nProc) % nProc;                
            }else{
                recv_chunk = (rank + 2 + i) % nProc;
                send_chunk = (rank + 1 + i) % nProc;
            }
            
            double* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

            MPI_Irecv(buffer, segment_sizes[recv_chunk],
                    datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

            MPI_Send(segment_send, segment_sizes[send_chunk],
                    MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);

            double *segment_update = &(output[segment_ends[recv_chunk] -
                                             segment_sizes[recv_chunk]]);

            // Wait for recv to complete before reduction
            MPI_Wait(&recv_req, &recv_status);

        //    reduce(segment_update, buffer, segment_sizes[recv_chunk]);
            reduce_local(buffer, segment_update, segment_sizes[recv_chunk], datatype, MPI_SUM);
        }
    }else{
        for (int i = 0; i < nProc - 1; ++i) {
            if(redscat == 2){
                recv_chunk = (rank - i - 1 + nProc) % nProc;
                send_chunk = (rank - i + nProc) % nProc;
            }else{
		recv_chunk = (rank + i + 1 + nProc) % nProc;
                send_chunk = (rank + i + nProc) % nProc;
            }
            // Segment to send - at every iteration we send segment (r+1-i)
            double* segment_send = &(output[segment_ends[send_chunk] -
                                           segment_sizes[send_chunk]]);

            // Segment to recv - at every iteration we receive segment (r-i)
            double* segment_recv = &(output[segment_ends[recv_chunk] -
                                           segment_sizes[recv_chunk]]);
            MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                    datatype, send_to, 0, segment_recv,
                    segment_sizes[recv_chunk], datatype, recv_from,
                    0, MPI_COMM_WORLD, &recv_status);
        }
    }
    delete [] buffer;
}

//void buckAllreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
void buckAllreduce(double *sendbuf, double *recvbuf, MPI_Aint count)
{    
    int size, myrank, x, y, z;
    int dimensions, dimensions_sizes[3];
    int relcoord[LDIM];
    int rc, outppn;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    rc = FJMPI_Topology_get_dimension(&dimensions);
    if (rc != FJMPI_SUCCESS) {
    	fprintf(stderr, "FJMPI_Topology_get_dims ERROR\n");
    	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    rc = FJMPI_Topology_get_shape(&x, &y, &z);
    if (rc != FJMPI_SUCCESS) {
    	fprintf(stderr, "FJMPI_Topology_get_shape ERROR\n");
    	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (dimensions == 1){
   // 	printf("My Shape: X= %d", x);
        dimensions_sizes[0] = x;
    }else if(dimensions == 2){
   //   printf("My Shape: X= %d", x);
        dimensions_sizes[0] = x;
   //     printf(", Y= %d", y);
        dimensions_sizes[1] = y;
    }else if(dimensions == 3){
    //   printf("My Shape: X= %d", x);
	dimensions_sizes[0] = x;
    //    printf(", Y= %d", y);
        dimensions_sizes[1] = y;
    //    printf(", Z= %d", z);
        dimensions_sizes[2] = z;
    }else{
        fprintf(stderr, "%d dimensions not supported.", dimensions);
        exit(-1);
    }
    
    double *data = new double[count];
    memcpy((void*) data, (void*) sendbuf, count * sizeof(double));
    
 // Adjust count to the number of data sets
    while(count % (2*dimensions)){
       count += 1;
    }

 // Get the neighbor nodes on each dimension
    int recvfrom[LDIM];
    int sendto[LDIM];
   // getCoordFromId(myrank, dimensions_sizes, dimensions, relcoord, size);
    rc = FJMPI_Topology_get_coords(MPI_COMM_WORLD, myrank, FJMPI_LOGICAL, dimensions, relcoord);
    if (rc != FJMPI_SUCCESS) {
   	fprintf(stderr, "FJMPI_Topology_get_coords ERROR\n");
    	MPI_Abort(MPI_COMM_WORLD, 1);
    }
  /*  switch(dimensions) {
        case 1:
            printf("rank to x : rank= %d, (X)=( %d ) \n",myrank, relcoord[0]);
            break;
        case 2:
            printf("rank to xy : rank= %d, (X,Y)=( %d, %d ) ",myrank, relcoord[0], relcoord[1]);
            break;
        case 3:
            printf(stderr, "rank to xyz : rank= %d, (X,Y,Z)=( %d, %d, %d ) \n", myrank, relcoord[0], relcoord[1], relcoord[2]);
            break;
        default:
            break;
        }
  */ 
    for(size_t i = 0; i < dimensions; i++){
        int coord[LDIM]={0,0,0};
        for(size_t j=0; j < dimensions; j++){ coord[j] = relcoord[j]; }
        // Next node
        coord[i] = (relcoord[i] + 1) % dimensions_sizes[i];
    //    sendto[i] = getIdFromCoord(coord, dimensions_sizes, dimensions);
        rc = FJMPI_Topology_get_ranks(MPI_COMM_WORLD, FJMPI_LOGICAL, coord, 1, &outppn, &sendto[i]);
        if (rc != FJMPI_SUCCESS) {
            fprintf(stderr, "FJMPI_Topology_get_ranks ERROR\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
          // Previous node
        coord[i] = (relcoord[i] - 1 + dimensions_sizes[i]) % dimensions_sizes[i];   //
    //    recvfrom[i] = getIdFromCoord(coord, dimensions_sizes, dimensions);
        rc = FJMPI_Topology_get_ranks(MPI_COMM_WORLD, FJMPI_LOGICAL, coord, dimensions, &outppn, &recvfrom[i]);
        if (rc != FJMPI_SUCCESS) {
            fprintf(stderr, "FJMPI_Topology_get_ranks ERROR\n");
           MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
 // Divide data into 2*num_dimensions datasets (buckets) and perform allreduce on each.
    size_t data_size[LDIM];
    size_t offsets[LDIM];
    for(int i = 0; i < dimensions; i++){
        data_size[i] = count/(2*dimensions);
        offsets[i] = 0;
    }    
    
    // Saving the offsets from reduce-scatter loops for corresponding allgather loops.
    size_t loop_offset_r[dimensions*dimensions];
    size_t loop_offset_l[dimensions*dimensions];
    size_t loop_data_size[dimensions*dimensions];

 // Reduce-scatter loops   
    for(int s = 0; s < dimensions; s++){
        for(int i = 0; i < dimensions; i++){
            int current_d = (i + s) % dimensions;
            size_t offset_l = (2*i)*(count/(2*dimensions))   + offsets[i];
            size_t offset_r = (2*i+1)*(count/(2*dimensions)) + offsets[i];
            assert(offset_l + data_size[i] <= (2*i+1)*(count/(2*dimensions)));
            
	        loop_offset_l[s*dimensions+i]= offset_l;
            loop_offset_r[s*dimensions+i]= offset_r;
            loop_data_size[s*dimensions+i]= data_size[i];

            ringRedScatAG(data + offset_l, data_size[i], dimensions_sizes[current_d], relcoord[current_d], recvfrom[current_d], sendto[current_d], 0);
            ringRedScatAG(data + offset_r, data_size[i], dimensions_sizes[current_d], relcoord[current_d], sendto[current_d], recvfrom[current_d], 1);
	    
            if(s != dimensions - 1){
                data_size[i] /= dimensions_sizes[current_d];
                offsets[i] += relcoord[current_d]*data_size[i];
            }
        }
   }
   
 // Allgather loops
    for(int s = dimensions - 1; s >= 0; s--){
        for(int i = dimensions - 1; i >= 0; i--){
            int current_d = (i + s) % dimensions;
            size_t offset_l = loop_offset_l[s*dimensions+i];
            size_t offset_r = loop_offset_r[s*dimensions+i];
            data_size[i] = loop_data_size[s*dimensions+i];
            assert(offset_l + data_size[i] <= (2*i+1)*(count/(2*dimensions)));
            
            ringRedScatAG(data + offset_l, data_size[i], dimensions_sizes[current_d], relcoord[current_d], recvfrom[current_d], sendto[current_d], 2);
            ringRedScatAG(data + offset_r, data_size[i], dimensions_sizes[current_d], relcoord[current_d], sendto[current_d], recvfrom[current_d], 3);
        }
    }
    memcpy((void*) recvbuf, (void*) data, count * sizeof(double));
    delete [] data;
}

    
/**
 * Program entry
 */
int main(int argc, char* argv[])
{
	int thisProc, nProc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisProc);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);

	int count = nProc*12;  // *6 //

	// initialize sendbuf
	double *sendbuf = new double[count];
	for (size_t i = 0; i < count; i++)
        sendbuf[i] = (i+1)*(thisProc+1);

    // initialize recvbuf
    double *recvbuf = new double[count];
    double *recvbufGround = new double[count];
    double *recvbuf_b = new double[count];
    
 // Ref. allreduce implementations 
    allreduce(sendbuf, recvbufGround, count);    // Default MPI_allreduce
    // allreduceRSAG(sendbuf, recvbuf, nProc);  // Reduce-scatter/allgather 
    allreduceRing(sendbuf, recvbuf, count);      // Topology un-aware Ring_allreduce  

    bool test = true;
    for (int i=0; i<count; i++){
//        printf("%f\n", recvbufGround[i]);
        if (recvbuf[i]!=recvbufGround[i])             test=false;
    }
    if (thisProc == 0){
        if (test)
            cout<<"Ring Allreduce Test passed"<<endl;
        else
            cout<<"Ring Allreduce Test failed"<<endl;
    }

 //  Test bucket allgorithm
    buckAllreduce(sendbuf, recvbuf_b, count);
    test = true;
    for (int i=0; i<count; i++){
 //       printf("reduced value:%f\t, error margin %f\n", recvbuf_b[i], recvbufGround[i] - recvbuf_b[i]);
        if (recvbufGround[i]!=recvbuf_b[i]) test=false ;
    }
    if (thisProc == 0){
        if (test)
            cout<<"Bucket Allreduce Test passed"<<endl;
        else
            cout<<"Bucket Allreduce Test failed"<<endl;
    }
	MPI_Finalize();
	exit(EXIT_SUCCESS);
}

