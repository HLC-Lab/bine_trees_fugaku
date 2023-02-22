#!/usr/bin/env python3
import math
import copy
import math
import numpy as np
import random

# Convert a rank id into a list of d-dimensional coordinates
def get_coord_from_id(id, dimensions):
    if len(dimensions) == 1:
        return [id]
    elif len(dimensions) == 2:
        return [id // dimensions[1], id % dimensions[1]]
    elif len(dimensions) == 3:
        return [(id // dimensions[1]) % dimensions[0], id % dimensions[1], id // (dimensions[0]*dimensions[1])]

# Convert d-dimensional coordinates into a rank id
def get_id_from_coord(coord, dimensions):
    if len(dimensions) == 1:
        return coord[0]
    elif len(dimensions) == 2:
        return coord[0]*dimensions[1] + coord[1]
    else:
        return int(coord[2]*(dimensions[0]*dimensions[1]) + get_id_from_coord(coord, dimensions[0:2]))

# Starting from the data of each node, computes the final data that we would expect after running an allreduce
def compute_expected_data(data, p, collective):
    expected_res = np.zeros((p, p))
    sum = np.zeros(p)
    for i in range(0, p):
        for j in range(0, p):
            sum[j] += data[i][j]
    if collective == "ALLREDUCE":
        # Compute expected result (Each row is a host)
        for i in range(0, p):            
            expected_res[i] = sum
    else: # REDUCESCATTER
        for i in range(0, p):            
            expected_res[i] = sum
            for j in range(0, p):
                if j != i:
                    expected_res[i][j] = 0
    return expected_res

# Gets the distance of the peer of rank 'id' at the step-th step on a dimension-dimensional network, considering the next_direction
def get_distance(id, step, next_direction, dimensions):
    distances = [1, 1, 3, 5, 11, 21, 43, 85] # Jacobsthal sequence!! (https://oeis.org/search?q=1%2C+1%2C+3%2C+5%2C+11%2C+21&language=english&go=Search)
    dim = step % len(dimensions)
    step_relative_to_dim = step // len(dimensions)
    return (next_direction[id][dim])*distances[step_relative_to_dim]

# Gets the peer of rank 'id' at the step-th step on a dimension-dimensional network, considering the next_direction
def get_peer(id, step, next_direction, dimensions):
    distance = get_distance(id, step, next_direction, dimensions)
    coord = get_coord_from_id(id, dimensions)
    target_dim =  step % len(dimensions) # TODO: Extend for the cases where we start from different dimensions (multiported)
    peer_coord = coord
    peer_coord[target_dim] = (peer_coord[target_dim] + distance) % dimensions[target_dim]
    return int(get_id_from_coord(peer_coord, dimensions))

def find_reducescatter_indexes(dimensions, starting_step, sender):
    # To have it more generic, I could just always assume sender 0
    # and then add the sender later.    
    p = 1
    for d in dimensions:
        p *= d
    steps = int(math.log2(p))
    d = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341]
    a = [-1]*p
    for step in range(starting_step, steps):
        # Direction is positive only when sender and step are both even, or when they are both odd.
        # In both cases their sum is going to be a positive number, and then the power will be a +1
        pos_or_neg = (-1)**(sender+step) 
        dest = (sender + pos_or_neg*d[step]) % len(a) 
        a[dest] = step
        sender = dest
        if step != steps - 1:
            for x in range(0, len(a)):
                if a[x] <= step and a[x] != -1:
                    # Direction is positive only when x and the next step (step+1) are both even, or when they are both odd.
                    # In both cases their sum is going to be a positive number, and then the power will be a +1
                    pos_or_neg = (-1)**(x+(step+1))
                    dest = (x + pos_or_neg*d[step + 1]) % len(a)
                    a[dest] = step + 1
    a = np.array(a)
    a[a >= 0] = 1
    a[a == -1] = 0
    #print("Step " + str(starting_step) + ": " + str(a))
    return a

def recv(dimensions, receiver, sender, data_new, data_old, collective, step):
    if collective == "ALLREDUCE":
        # Aggregate
        for k in range(0, len(data_new[receiver])):
            data_new[receiver][k] += data_old[sender][k]
    else: # REDUCESCATTER
        a = find_reducescatter_indexes(dimensions, step, sender)
        for k in range(0, len(data_new[receiver])):
            if a[k]:
                actual_block = k
                data_new[receiver][actual_block] += data_old[sender][actual_block]

#find_reducescatter_indexes(1024, 0, 0)


def run_collectives(dimensions, data, collective):
    # Algo:
    # 1. I take the tuple representing the coordinates
    # 2. Each node proceeds one dimension at a time (in a synchronized way, e.g. x first, then y, then z)
    # 3. Once a dimension is decided, the decision is whether to go on the positive or negative direction. 
    #    Let us suppose that in the first step everyone picks the x dimension. Even nodes on that dimension
    #    will send towards the positive direction, odd nodes towards the negative direction.
    p = 1
    for d in dimensions:
        p *= d

    next_direction = np.zeros((p, len(dimensions))) # For each dimension, each rank remembers on which direction it should send next time
    sum_of_distances = 0
    sum_of_distances_ref = 0
    bw_new = 0
    bw_ref = 0
    bw_ideal = 0

    # Set starting directions
    for i in range(0, p):
        coord = get_coord_from_id(i, dimensions)
        for c in range(0, len(coord)):
            if coord[c] % 2 == 0:
                next_direction[i][c] = 1
            else:
                next_direction[i][c] = -1

    for step in range(0, int(math.log2(p))):        
        data_c = copy.deepcopy(data)
        # Bit array to check that no one sends the same data to more than one node
        recv_from = np.zeros(p)
        for rank in range(0, p):                   
            peer = get_peer(rank, step, next_direction, dimensions)         
            next_direction[rank][step % len(dimensions)] *= -1
            # Check that each node sends at most at one node
            if recv_from[peer]:
                print("Double recv")
                exit(-1)
            #print("Step " + str(i) + ": " + str(j) + "->" + str(peer))
            recv_from[peer] = 1
            recv(dimensions, rank, peer, data_c, data, collective, step)
        # Check that everyone sent something
        for k in range(0, p):
            if not recv_from[k]:
                print("Nosent")
                exit(-1)

        # Compute some stats
        distance = abs(get_distance(0, step, next_direction, dimensions))
        sum_of_distances += distance         
        bw_ideal += (1/2**(step+1)) 
        bw_new += (1/2**(step+1))*distance
        sum_of_distances_ref += 2**(step / len(dimensions))
        bw_ref += (1/2**(step+1))*2**(step / len(dimensions))
        data = copy.deepcopy(data_c)
    return (data, sum_of_distances, sum_of_distances_ref, bw_new, bw_ref, bw_ideal)

def main():
    #collective = "ALLREDUCE"
    collective = "REDUCESCATTER"
    #dimensions = [32, 32]
    dimensions = [64]
    p = 1
    for d in dimensions:
        p *= d

    # Initialize random starting data
    data = np.zeros((p, p))
    for i in range(0, p):
        for j in range(0, p):
            data[i][j] = random.randint(0, 100)
    expected_res = compute_expected_data(data, p, collective)

    data, sum_of_distances, sum_of_distances_ref, bw_new, bw_ref, bw_ideal = run_collectives(dimensions, data, collective)
    # Check correctness of the result
    # For reduce scatter, clean the partially reduced data
    if collective == "REDUCESCATTER":
        for i in range(0, p):
            for j in range(0, p):
                if i != j:
                    data[i][j] = 0

    for i in range(0, p):
        for j in range(0, p):
            if data[i][j] != expected_res[i][j]:
                print("FAIL!")
                #print(data)
                #print(expected_res)
                exit(1)
    
    print("Sum of distances: " + str(sum_of_distances) + " ref: " + str(sum_of_distances_ref))
    print("Bw new: " + str(bw_new) + " bw ref: " + str(bw_ref) + " bw ideal: " + str(bw_ideal))

if __name__ == "__main__":
    main()

