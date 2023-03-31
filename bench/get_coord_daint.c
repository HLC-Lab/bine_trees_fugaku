#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv){
    char* coordfile = "/proc/cray_xt/cname";
    FILE * f = fopen(coordfile, "r");
    if (f==NULL) { printf("Error opening %s\n", coordfile); return 0; }
    /* cabinet, row, chassis, slot, node */
    int cabinet, row, chassis, slot, node;
    if(fscanf(f, "c%i-%ic%is%in%i", &cabinet, &row, &chassis, &slot, &node) != 5){
      fclose(f);
      return 1;
    }
    fclose(f);
    char hostname[1024];
    gethostname(hostname, 1024);
    printf("Hostname: %s Cabinet: %d Row: %d Chassis: %d Slot: %d Node: %d\n", hostname, cabinet, row, chassis, slot, node);
    return 0;
}