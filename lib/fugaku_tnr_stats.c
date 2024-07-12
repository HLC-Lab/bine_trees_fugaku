// Adapted from https://www.fugaku.r-ccs.riken.jp/doc_root/en/user_guides/use_latest/JobExecution/TofuStatistics.html
// More info on the reported stats at https://www.fugaku.r-ccs.riken.jp/doc_root/en/user_guides/use_latest/JobExecution/TofuStatistics.html
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>

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


char* port_name[NUM_TNR] = {"A", "C", "B-", "B+", "X-", "X+", "Y-", "Y+", "Z-", "Z+" };
void print_tnr_stats(uint64_t reading[NUM_TNR][PA_LEN], uint rank, FILE* stream){
  //fprintf(stream, "rank,port_name,pa[0],pa[1],pa[2],pa[3],pa[6],pa[7],pa[16],pa[17]\n");
  fprintf(stream, "rank,port_name,zero_credit_cycles_vc0,zero_credit_cycles_vc1,zero_credit_cycles_vc2,zero_credit_cycles_vc3,sent_pkts,sent_bytes,recvd_pkts,recvd_bytes\n");

  for(port_no = 0; port_no < NUM_TNR; port_no++){
    fprintf(stream, "%d,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
	    rank,port_name[port_no], reading[port_no][0], reading[port_no][1], reading[port_no][2], reading[port_no][3], reading[port_no][6], reading[port_no][7]*16, reading[port_no][16], reading[port_no][17]*16);
  }
}


/*
int main(int argc, char *argv[]) {
	int port_no, ret;
	uint64_t pa[PA_LEN];

	printf("port_name,pa[0],pa[1],pa[2],pa[3],pa[6],pa[7],pa[16],pa[17]\n");
	for(port_no = 1; port_no <= 10; port_no++) {
		ret = port_stat_ioctl(port_no, pa);
		if (ret < 0)
			return ret;
		printf("%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
			port_name[port_no], pa[0], pa[1], pa[2], pa[3], pa[6], pa[7], pa[16], pa[17]);
	}

	return 0;
}
*/

