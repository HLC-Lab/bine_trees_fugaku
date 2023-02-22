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

# Starting from the data of each node, computes the final data that we would expect after running a collective
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
    elif collective == "REDUCESCATTER":
        for i in range(0, p):            
            expected_res[i] = sum
            for j in range(0, p):
                if j != i:
                    expected_res[i][j] = 0
    else: # ALLGATHER
        for i in range(0, p):
            for j in range(0, p):
                expected_res[i][j] = data[j][j]
    return expected_res

# Gets the distance of the peer of rank 'sender' at the step-th step on a dimension-dimensional network, considering the next_direction
def get_peer_distance(sender, step, dimensions, collective):
    distances = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341] # Jacobsthal sequence!! (https://oeis.org/search?q=1%2C+1%2C+3%2C+5%2C+11%2C+21&language=english&go=Search)
    dim = step % len(dimensions)
    step_relative_to_dim = step // len(dimensions)
    # Derive next_direction starting from sender, step, dimensions
    coord = get_coord_from_id(sender, dimensions)
    # In which direction did I start on the target dimension?
    # If my coordinate on that dimension is even, then it started from positive
    # otherwise from negative
    if coord[dim] % 2 == 0:
        next_direction = (-1)**step_relative_to_dim
    else:
        next_direction = -1 * ((-1)**step_relative_to_dim)
    if collective == "ALLGATHER": # Distance "halving"
        return next_direction*distances[int(math.log2(dimensions[0])) - step_relative_to_dim - 1]
    else: # Distance "doubling"
        return next_direction*distances[step_relative_to_dim]

# Gets the peer of rank 'sender' at the step-th step on a dimension-dimensional network, considering the next_direction
def get_peer(sender, step, dimensions, collective):
    distance = get_peer_distance(sender, step, dimensions, collective)
    coord = get_coord_from_id(sender, dimensions)
    target_dim =  step % len(dimensions) # TODO: Extend for the cases where we start from different dimensions (multiported)
    peer_coord = coord
    peer_coord[target_dim] = (peer_coord[target_dim] + distance) % dimensions[target_dim]
    return int(get_id_from_coord(peer_coord, dimensions))

# Returns a list of indexes of the data to be sent in a reduce-scatter collective
# by 'sender' at step 'step' on a torus with 'dimensions' dimensions.
def find_reducescatter_indexes(sender, step, dimensions):
    p = 1
    for d in dimensions:
        p *= d
    if step >= math.log2(p): # Base case
        return []
    else:
        l = []
        for s in range(step, int(math.log2(p))):
            peer = get_peer(sender, s, dimensions, "REDUCESCATTER")
            l += [peer] # I add all the nodes that I directly reach ...
            l += find_reducescatter_indexes(peer, s + 1, dimensions) # ... plus all the nodes that those nodes reach.
        return l 

# Returns a list of indexes of the data to be sent in an allgather collective
# by 'sender' at step 'step' on a torus with 'dimensions' dimensions.
def find_allgather_indexes(sender, step, dimensions):
    if step == 0: # Base case
        return [sender]
    else:
        # I send whatever I sent in the previous step ...
        l = find_allgather_indexes(sender, step - 1, dimensions)
        # ... plus what I received in the previous step
        peer = get_peer(sender, step - 1, dimensions, "ALLGATHER")
        l += find_allgather_indexes(peer, step - 1, dimensions)
        return l 

# Moves data from sender to receiver
def send(sender, step, dimensions, receiver, data_old, data_new, collective):
    if collective == "ALLREDUCE":
        # Aggregate
        for k in range(0, len(data_new[receiver])):
            data_new[receiver][k] += data_old[sender][k]
    elif collective == "REDUCESCATTER":
        a = find_reducescatter_indexes(sender, step, dimensions)
        for k in a:
            data_new[receiver][k] += data_old[sender][k]
    else: # ALLGATHER
        # I send whathever I gathered so far to the next node
        # Thus I need to know what the sender received up to this point
        a = find_allgather_indexes(sender, step, dimensions)
        for k in a:
            data_new[receiver][k] = data_old[sender][k]

# Run a collective 'collective' on the data 'data' on a torus with 'dimensions' dimensions.
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

    sum_of_distances = 0
    sum_of_distances_ref = 0
    bw_new = 0
    bw_ref = 0
    bw_ideal = 0

    for step in range(0, int(math.log2(p))):        
        data_c = copy.deepcopy(data)
        # Bit array to check that no one sends the same data to more than one node
        recv_from = np.zeros(p)
        for rank in range(0, p):             
            peer = get_peer(rank, step, dimensions, collective)         
            # Check that each node sends at most to one node
            if recv_from[peer]:
                print("Double recv")
                exit(-1)
            #print("Step " + str(i) + ": " + str(j) + "->" + str(peer))
            recv_from[peer] = 1
            # If I recv from peer, peer is sending to me
            send(peer, step, dimensions, rank, data, data_c, collective)
        # Check that everyone sent something
        for k in range(0, p):
            if not recv_from[k]:
                print("Nosent")
                exit(-1)

        # Compute some stats
        distance = abs(get_peer_distance(0, step, dimensions, collective))
        sum_of_distances += distance                 
        if collective == "ALLGATHER": # Distance halving
            distance_ref = 2**((math.log2(p) - step - 1) / len(dimensions))
            bw_ideal += 2**step
            bw_new += (2**step)*distance
            bw_ref += (2**step)*distance_ref
        else: # Distance doubling
            distance_ref = 2**(step / len(dimensions))
            bw_ideal += (1/2**(step+1)) 
            bw_new += (1/2**(step+1))*distance
            bw_ref += (1/2**(step+1))*distance_ref
        sum_of_distances_ref += distance_ref
        data = copy.deepcopy(data_c)
    return (data, sum_of_distances, sum_of_distances_ref, bw_new, bw_ref, bw_ideal)

def main():
    #collective = "ALLREDUCE"
    #collective = "REDUCESCATTER"
    collective = "ALLGATHER"
    #dimensions = [8, 8, 8]
    dimensions = [8, 8]
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

    # Check correctness
    for i in range(0, p):
        for j in range(0, p):
            if data[i][j] != expected_res[i][j]:
                print("FAIL!")
                print(data)
                print(expected_res)
                exit(1)
    
    #print("Sum of distances: " + str(sum_of_distances) + " ref: " + str(sum_of_distances_ref))
    #print("Bw new: " + str(bw_new) + " bw ref: " + str(bw_ref) + " bw ideal: " + str(bw_ideal))
    # bw is not the bandwidth but the bandwidth term (i.e., higher=worst), so we invert
    normalized_bw_new = 1.0 / (bw_new / bw_ideal)
    normalized_bw_ref = 1.0 / (bw_ref / bw_ideal)
    normalized_bw_new_wrt_ref = normalized_bw_new / normalized_bw_ref
    print(f"bw new vs ideal: {normalized_bw_new} bw ref vs ideal: {normalized_bw_ref} bw new vs ref: {normalized_bw_new_wrt_ref}")

if __name__ == "__main__":
    main()

