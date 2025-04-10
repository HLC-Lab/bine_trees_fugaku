#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cinttypes>

#ifdef FUGAKU
// Adapted from https://www.fugaku.r-ccs.riken.jp/doc_root/en/user_guides/use_latest/JobExecution/TofuStatistics.html
// More info on the reported stats at https://www.fugaku.r-ccs.riken.jp/doc_root/en/user_guides/use_latest/JobExecution/TofuStatistics.html
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>

int MPI_Bcast_f(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm );

#define TOFU_DEV_INFO "/proc/tofu/dev/info"
#define TOF_IOCTL_GET_PORT_STAT _IOWR('d', 9, long)
#define PA_LEN 31
#define NUM_TNR 10
#define IOCTL_REQ_MASK 0xFFFCFF30

struct tof_get_port_stat {
        int port_no;
        uint64_t mask;
        uint64_t pa[PA_LEN];
};

int port_stat_ioctl(int port_no, uint64_t *pa) {
        int ret = 0, fd;
        struct tof_get_port_stat req;

        fd = open(TOFU_DEV_INFO, O_RDWR|O_CLOEXEC);
        if (fd < 0) {
                perror("open(TOFU_DEV_INFO)");
                return -1;
        }

        req.port_no = port_no;
        req.mask = IOCTL_REQ_MASK;
        memset(req.pa, 0, sizeof(req.pa));

        ret = ioctl(fd, TOF_IOCTL_GET_PORT_STAT, &req);
        if (ret < 0) {
                perror("ioctl(TOF_IOCTL_GET_PORT_STAT)");
        } else {
                memcpy(pa, req.pa, sizeof(req.pa));
        }

        close(fd);
        return ret;
}

int read_tnr_stats(uint64_t reading[NUM_TNR][PA_LEN]){
        int port_no, ret;
        uint64_t pa[PA_LEN];
        for(port_no = 1; port_no <= NUM_TNR; port_no++) {
                ret = port_stat_ioctl(port_no, pa);
                if (ret < 0)
                        return ret;
                memcpy(reading[port_no - 1], pa, sizeof(pa));
        }

        return 0;
}

void diff_tnr_stats(uint64_t start[NUM_TNR][PA_LEN], uint64_t stop[NUM_TNR][PA_LEN], uint64_t diff[NUM_TNR][PA_LEN]){
  int port_no, pa;
  for(port_no = 0; port_no < NUM_TNR; port_no++){
    for(pa = 0; pa < PA_LEN; pa++){
      diff[port_no][pa] = stop[port_no][pa] - start[port_no][pa];
    }
  }
}


char port_name[NUM_TNR][3] = {"A", "C", "B-", "B+", "X-", "X+", "Y-", "Y+", "Z-", "Z+" };
void print_tnr_stats(uint64_t reading[NUM_TNR][PA_LEN], uint rank){
  char filename[100];
  snprintf(filename, sizeof(filename), "tnr_stats_%d.csv", rank);  
  FILE *f = fopen(filename, "w");
  fprintf(f, "rank,port_name,zero_credit_cycles_vc0,zero_credit_cycles_vc1,zero_credit_cycles_vc2,zero_credit_cycles_vc3,sent_pkts,sent_bytes,recvd_pkts,recvd_bytes\n");
  int port_no;
  for(port_no = 0; port_no < NUM_TNR; port_no++){
    fprintf(f, "%d,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
            rank,port_name[port_no], reading[port_no][0], reading[port_no][1], reading[port_no][2], reading[port_no][3], reading[port_no][6], reading[port_no][7]*16, reading[port_no][16], reading[port_no][17]*16);
  }
  fflush(f);
  fclose(f);
}
#endif

void voidop(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
    return;
}

typedef enum{
    RUN_TYPE_VALIDATION = 0,
    RUN_TYPE_BENCHMARK = 1
}RunType;

int run_collective(RunType rt, const char* collective, const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, size_t size){
    int r = MPI_SUCCESS;
    if(!strcmp(collective, "MPI_Allreduce")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Allreduce(sendbuf, recvbuf, count, dt, op, MPI_COMM_WORLD);
        }else{
            r = MPI_Allreduce(sendbuf, recvbuf, count, dt, op, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Reduce_scatter")){
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        for(size_t i = 0; i < size; i++){
            recvcounts[i] = count;
        }
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, dt, op, MPI_COMM_WORLD);
        }else{
            r = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, dt, op, MPI_COMM_WORLD);
        }
        free(recvcounts);
    }else if(!strcmp(collective, "MPI_Allgather")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Allgather(sendbuf, count, dt, recvbuf, count, dt, MPI_COMM_WORLD);
        }else{
            r = MPI_Allgather(sendbuf, count, dt, recvbuf, count, dt, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Bcast")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Bcast((void*) sendbuf, count, dt, 0, MPI_COMM_WORLD);
        }else{
            r = MPI_Bcast_f((void*) sendbuf, count, dt, 0, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Alltoall")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Alltoall(sendbuf, count, dt, recvbuf, count, dt, MPI_COMM_WORLD);
        }else{
            r = MPI_Alltoall(sendbuf, count, dt, recvbuf, count, dt, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Scatter")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Scatter(sendbuf, count, dt, recvbuf, count, dt, 0, MPI_COMM_WORLD);
        }else{
            r = MPI_Scatter(sendbuf, count, dt, recvbuf, count, dt, 0, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Gather")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Gather(sendbuf, count, dt, recvbuf, count, dt, 0, MPI_COMM_WORLD);
        }else{
            r = MPI_Gather(sendbuf, count, dt, recvbuf, count, dt, 0, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Reduce")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Reduce(sendbuf, recvbuf, count, dt, op, 0, MPI_COMM_WORLD);
        }else{
            r = MPI_Reduce(sendbuf, recvbuf, count, dt, op, 0, MPI_COMM_WORLD);
        }
    }

    /*
    char val[MPI_MAX_INFO_VAL];
    int flag = 0;
    MPI_Info info;
    MPI_Comm_get_info(MPI_COMM_WORLD, &info);
    MPI_Info_get(info, "last_allreduce_algorithm", MPI_MAX_INFO_VAL, val, &flag);
    if(flag){//flag is 1, Info is obtained
      fprintf(stderr, "val is %s\n",val);//(2)
    }
    MPI_Info_free(&info);
    */
    return r;
}

static inline void allocate_buffers(const char* collective, size_t count, size_t size, size_t dtsize, char** sendbuf, char** recvbuf, char** recvbuf_validation, int rank){
    size_t send_count, recv_count;
    if(!strcmp(collective, "MPI_Allreduce")){
        send_count = count;
        recv_count = count;
    }else if(!strcmp(collective, "MPI_Bcast")){
        send_count = count;
        recv_count = count;
    }else if(!strcmp(collective, "MPI_Reduce")){
        send_count = count;
        recv_count = count;
    }else if(!strcmp(collective, "MPI_Reduce_scatter")){
        send_count = count*size;
        recv_count = count;
    }else if(!strcmp(collective, "MPI_Allgather")){
        send_count = count;
        recv_count = count*size;
    }else if(!strcmp(collective, "MPI_Scatter")){
        send_count = count*size;
        recv_count = count;
    }else if(!strcmp(collective, "MPI_Gather")){
        send_count = count;
        recv_count = count*size;
    }else if(!strcmp(collective, "MPI_Alltoall")){
        send_count = count*size;
        recv_count = count*size;
    }else{
        fprintf(stderr, "Unknown collective %s\n", collective);
        exit(-1);
    }
    *sendbuf = (char*) malloc(dtsize*send_count);    
    *recvbuf = (char*) malloc(dtsize*recv_count);
    *recvbuf_validation = (char*) malloc(dtsize*recv_count); 

    // Initialize sendbuf with random values
    // For ranks != root (i.e., rank 0 in our case), sendbuf only makes sense for collectives different from bcast and scatter
    if((rank == 0) || (strcmp(collective, "MPI_Bcast") && strcmp(collective, "MPI_Scatter"))){
        for(size_t i = 0; i < dtsize*send_count; i++){
            (*sendbuf)[i] = rand() % 1024;
        }        
    }else{
        memset(*sendbuf, 0, dtsize*send_count);
    }
}


// Usage: ./bench collective type msgsize(elems) iterations
int main(int argc, char** argv){
    int warmup = 10;    
    char* collective = argv[1];
    char* type = argv[2];
    int count = atoi(argv[3]);
    int iterations = atoi(argv[4]);
    double* samples = (double*) malloc(sizeof(double)*iterations);
    double* samples_all = NULL;
    long i;
    int r;
    MPI_Init(&argc, &argv);
    int rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    

    MPI_Op MPI_VOIDOP;
    MPI_Op_create(voidop, 1, &MPI_VOIDOP);
    MPI_Datatype dt;
    MPI_Op op = MPI_SUM;
    if(strcmp(type, "CHAR") == 0){
        dt = MPI_CHAR;
    }else if(strcmp(type, "BYTE") == 0){
        dt = MPI_BYTE;
    }else if(strcmp(type, "FLOAT") == 0){
        dt = MPI_FLOAT;
    }else if(strcmp(type, "INT") == 0){
        dt = MPI_INT;
    }else if(strcmp(type, "INT32") == 0){
        dt = MPI_INT32_T;
    }else if(strcmp(type, "VOID") == 0){
        dt = MPI_INT32_T;
        op = MPI_VOIDOP;
    }else{
        fprintf(stderr, "Unknown type %s\n", type);
        return 1;
    }
    int dtsize;
    MPI_Type_size(dt, &dtsize);
    srand(time(NULL)*rank);
    if(rank == 0){
        samples_all = (double*) malloc(sizeof(double)*iterations);
    }

    char *sendbuf, *recvbuf, *recvbuf_validation; // To check correctness of results
    allocate_buffers(collective, count, comm_size, dtsize, &sendbuf, &recvbuf, &recvbuf_validation, rank);
    r = run_collective(RUN_TYPE_VALIDATION, collective, sendbuf, recvbuf_validation, count, dt, op, comm_size);
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Rank %d: Validation failed with error %d\n", rank, r);
        return 1;
    }
    if(!strcmp(collective, "MPI_Bcast")){
        memcpy(recvbuf_validation, sendbuf, dtsize*count);
        if(rank != 0){
            memset(sendbuf, 0, dtsize*count);   
        }
    }

#ifdef FUGAKU
    //uint64_t tnr_start[NUM_TNR][PA_LEN];
    //uint64_t tnr_stop[NUM_TNR][PA_LEN];
    //uint64_t tnr_diff[NUM_TNR][PA_LEN];
    //assert(read_tnr_stats(tnr_start)==0);
#endif

    for(i = -warmup; i < iterations; i++){
        //usleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

//        if(rank == 0){
//            #define INTERVAL 1000 // Microseconds
//            struct timespec ts;
//            clock_gettime(CLOCK_MONOTONIC, &ts);
//            long current_us = (ts.tv_sec * 1000000L + ts.tv_nsec / 1000L);
//            while (1) {
//                clock_gettime(CLOCK_MONOTONIC, &ts);            
//                if ((ts.tv_sec * 1000000L + ts.tv_nsec / 1000L) - current_us > INTERVAL) {
//                    break;  // Exit the loop when the condition is met
//                }
//            }        
//        }

        double start_time = MPI_Wtime();
        // Run the collective
        r = run_collective(RUN_TYPE_BENCHMARK, collective, sendbuf, recvbuf, count, dt, op, comm_size);
        if(r != MPI_SUCCESS){
            fprintf(stderr, "Rank %d: Benchmark failed with error %d\n", rank, r);
            return 1;
        }

        if(i >= 0){
            samples[i] = (MPI_Wtime() - start_time);
        }
    }
    if(!strcmp(collective, "MPI_Bcast")){
        memcpy(recvbuf, sendbuf, dtsize*count);
    }

    size_t final_buffer_count = count;
    if(!strcmp(collective, "MPI_Alltoall") || !strcmp(collective, "MPI_Gather")  || !strcmp(collective, "MPI_Allgather")){
        final_buffer_count *= comm_size;
    }

    // Check correctness of results
    
    if((!strcmp(collective, "MPI_Gather") || !strcmp(collective, "MPI_Reduce")) && rank != 0){
        // On MPI_Gather and MPI_Reduce only rank 0 receives the result
    }else{
        char* skip_validation = getenv("LIBSWING_SKIP_VALIDATION");
        if(skip_validation == NULL || strcmp(skip_validation, "0") == 0){
            // Don't validate for VOID op
            if(strcmp(type, "VOID")){
                for(i = 0; i < dtsize*final_buffer_count; i++){
                    if(recvbuf[i] != recvbuf_validation[i]){
                        fprintf(stderr, "Rank %d: Validation failed at index %ld: %d != %d\n", rank, i, recvbuf[i], recvbuf_validation[i]);
                        return 1;
                    }
                }
            }
        }
        /**
        printf("Rank %d Sendbuf: ", rank);
        for(i = 0; i < dtsize*final_buffer_count; i++){
            printf("%d ", sendbuf[i]);
        }
        printf("\n");

        printf("Rank %d Recvbuf: ", rank);
        for(i = 0; i < dtsize*final_buffer_count; i++){
            printf("%d ", recvbuf[i]);
        }
        printf("\n");

        printf("Rank %d  Valbuf: ", rank);
        for(i = 0; i < dtsize*final_buffer_count; i++){
            printf("%d ", recvbuf_validation[i]);
        }
        printf("\n");
        **/
    }
#ifdef FUGAKU
    //assert(read_tnr_stats(tnr_stop)==0);
    //diff_tnr_stats(tnr_start, tnr_stop, tnr_diff);
    //print_tnr_stats(tnr_diff, rank);
#endif    


    //for(i = 0; i < iterations; i++){
    //    printf("%f\n", samples[i]);
    //}

    PMPI_Reduce(samples, samples_all, iterations, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){ 
        printf("highest\n");
        double avg_iteration = 0.0;
        for(i = 0; i < iterations; i++){
            //samples_all[i] /= comm_size;
            printf("%" PRId64"\n", (int64_t)(samples_all[i] * 1e9));
            avg_iteration += samples_all[i];
        }
        avg_iteration /= iterations;
        printf("Average runtime: %f\n", avg_iteration);
    }
    MPI_Finalize();
    free(sendbuf);
    if(recvbuf != sendbuf){
        free(recvbuf);
    }
    free(samples);
    return 0;
}
