#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define LIBSWING_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
static int smallest_negabinary[LIBSWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBSWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

// https://stackoverflow.com/questions/37637781/calculating-the-negabinary-representation-of-a-given-number-without-loops
static inline uint32_t binary_to_negabinary(int32_t bin) {
    assert(bin <= 0x55555555);
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}

static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t get_rank_negabinary_representation(uint32_t num_ranks, uint32_t rank){
    binary_to_negabinary(rank);
    uint32_t nba = -1, nbb = -1;
    size_t num_bits = ceil(log2(num_ranks));
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = binary_to_negabinary(rank);
        }
        if(in_range(rank - num_ranks, num_bits)){
            nbb = binary_to_negabinary(rank - num_ranks);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = binary_to_negabinary(-rank);
        }
        if(in_range(-rank + num_ranks, num_bits)){
            nbb = binary_to_negabinary(-rank + num_ranks);
        }
    }
    assert(nba != -1 || nbb != -1);

    if(nba == -1 && nbb != -1){
        return nbb;
    }else if(nba != -1 && nbb == -1){
        return nba;
    }else{ // Check MSB
        if(nba & (80000000 >> (32 - num_bits))){
            return nba;
        }else{
            return nbb;
        }
    }
}

int main(int argc, char** argv){
    uint32_t num_ranks = atoi(argv[1]);
    uint32_t rank = atoi(argv[2]);    
    uint32_t remap_rank = get_rank_negabinary_representation(num_ranks, rank);
    printf("%d\n", remap_rank);
    remap_rank = remap_rank ^ (remap_rank >> 1);
    printf("%d\n", remap_rank);
    size_t num_bits = ceil(log2(num_ranks));
    remap_rank = reverse(remap_rank) >> (32 - num_bits);
    printf("rank: %u, remap_rank: %u\n", rank, remap_rank);
    return 0;
}