#!/usr/bin/env python3
import math
import copy
import math
import numpy as np
import random
import argparse
import itertools

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
def get_peer_distance(sender, step, dimensions, collective, distances):
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
def get_peer(sender, step, dimensions, collective, distances):
    distance = get_peer_distance(sender, step, dimensions, collective, distances)
    coord = get_coord_from_id(sender, dimensions)
    target_dim =  step % len(dimensions) # TODO: Extend for the cases where we start from different dimensions (multiported)
    peer_coord = coord
    peer_coord[target_dim] = (peer_coord[target_dim] + distance) % dimensions[target_dim]
    return int(get_id_from_coord(peer_coord, dimensions))

# Returns a list of indexes of the data to be sent in a reduce-scatter collective
# by 'sender' at step 'step' on a torus with 'dimensions' dimensions.
def find_reducescatter_indexes(sender, step, dimensions, distances):
    p = 1
    for d in dimensions:
        p *= d
    if step >= math.log2(p): # Base case
        return []
    else:
        l = []
        for s in range(step, int(math.log2(p))):
            peer = get_peer(sender, s, dimensions, "REDUCESCATTER", distances)
            l += [peer] # I add all the nodes that I directly reach ...
            l += find_reducescatter_indexes(peer, s + 1, dimensions, distances) # ... plus all the nodes that those nodes reach.
        return l 

# Returns a list of indexes of the data to be sent in an allgather collective
# by 'sender' at step 'step' on a torus with 'dimensions' dimensions.
def find_allgather_indexes(sender, step, dimensions, distances):
    if step == 0: # Base case
        return [sender]
    else:
        # I send whatever I sent in the previous step ...
        l = find_allgather_indexes(sender, step - 1, dimensions)
        # ... plus what I received in the previous step
        peer = get_peer(sender, step - 1, dimensions, "ALLGATHER", distances)
        l += find_allgather_indexes(peer, step - 1, dimensions)
        return l 

# Moves data from sender to receiver
def send(sender, step, dimensions, receiver, data_old, data_new, collective, distances):
    if collective == "ALLREDUCE":
        # Aggregate
        for k in range(0, len(data_new[receiver])):
            data_new[receiver][k] += data_old[sender][k]
    elif collective == "REDUCESCATTER":
        a = find_reducescatter_indexes(sender, step, dimensions, distances)
        for k in a:
            data_new[receiver][k] += data_old[sender][k]
    else: # ALLGATHER
        # I send whathever I gathered so far to the next node
        # Thus I need to know what the sender received up to this point
        a = find_allgather_indexes(sender, step, dimensions, distances)
        for k in a:
            data_new[receiver][k] = data_old[sender][k]

# Run a collective 'collective' on the data 'data' on a torus with 'dimensions' dimensions.
def run_collectives(dimensions, data, collective, distances):
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
            peer = get_peer(rank, step, dimensions, collective, distances)         
            # Check that each node sends at most to one node
            if recv_from[peer]:
                print("Double recv")
                exit(-1)
            #print("Step " + str(step) + ": " + str(rank) + "->" + str(peer) + " (distance " + str(distances[step]) + ")")
            recv_from[peer] = 1
            # If I recv from peer, peer is sending to me
            send(peer, step, dimensions, rank, data, data_c, collective, distances)
        # Check that everyone sent something
        for k in range(0, p):
            if not recv_from[k]:
                print("Nosent")
                exit(-1)

        # Compute some stats
        distance = abs(get_peer_distance(0, step, dimensions, collective, distances))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collective", help="Collective to simulate [ALLREDUCE|REDUCESCATTER|ALLGATHER].", default="ALLREDUCE")
    parser.add_argument("-d", "--dimensions", help="Torus dimensions [AxBxCx...xZ].", default="8x8")
    parser.add_argument("-t", "--distances", help="Array of distances [default: Jacobsthal Sequence].", default=None)
    args = parser.parse_args()
    dimensions = [int(d) for d in args.dimensions.split("x")]
    p = 1
    for d in dimensions:
        p *= d

    try_all_distances = False
    distances_index = 0
    max_normalized_bw = 0
    max_normalized_bw_distances = []
    if not args.distances:
        distances = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341]
    elif args.distances == "ALL":
        try_all_distances = True
        tmp = []
        for t in range(0, int(p**(1/len(dimensions)))):
            tmp += [t]
        distances_it = itertools.product(tmp, repeat=int(math.log2(p)) // len(dimensions))
        distances = next(distances_it)
    else:
        distances = [int(d) for d in args.distances.split(",")]


    while True:
        # Initialize random starting data
        data = np.zeros((p, p))
        for i in range(0, p):
            for j in range(0, p):
                data[i][j] = random.randint(0, 100)
        expected_res = compute_expected_data(data, p, args.collective)

        data, sum_of_distances, sum_of_distances_ref, bw_new, bw_ref, bw_ideal = run_collectives(dimensions, data, args.collective, distances)

        # Compute stats
        #print("Sum of distances: " + str(sum_of_distances) + " ref: " + str(sum_of_distances_ref))
        #print("Bw new: " + str(bw_new) + " bw ref: " + str(bw_ref) + " bw ideal: " + str(bw_ideal))
        # bw is not the bandwidth but the bandwidth term (i.e., higher=worst), so we invert
        if bw_new:
            normalized_bw_new = 1.0 / (bw_new / bw_ideal)
            normalized_bw_ref = 1.0 / (bw_ref / bw_ideal)
            normalized_bw_new_wrt_ref = normalized_bw_new / normalized_bw_ref
            if not try_all_distances:
                print(f"bw new vs ideal: {normalized_bw_new} bw ref vs ideal: {normalized_bw_ref} bw new vs ref: {normalized_bw_new_wrt_ref}")
        else:
            normalized_bw_new = 0

        # Check correctness of the result
        # For reduce scatter, clean the partially reduced data
        if args.collective == "REDUCESCATTER":
            for i in range(0, p):
                for j in range(0, p):
                    if i != j:
                        data[i][j] = 0

        # Check correctness
        failed = False
        for i in range(0, p):
            for j in range(0, p):
                if data[i][j] != expected_res[i][j]:
                    if not try_all_distances:                    
                        print("FAIL!")
                        print(data)
                        print(expected_res)
                        exit(1)
                    else:
                        failed = True
        
        if not try_all_distances:            
            break
        else:
            if not failed:
                #print("Valid distance vector: " + str(distances))
                if normalized_bw_new >= max_normalized_bw:
                    max_normalized_bw = normalized_bw_new
                    if normalized_bw_new == max_normalized_bw:
                        max_normalized_bw_distances += [distances]
                    else:
                        max_normalized_bw_distances = [distances]
            # Generate next distances (if needed)
            try:
                distances = next(distances_it)
            except StopIteration as e:
                break

    if try_all_distances:
        for d in max_normalized_bw_distances:
            print(f"Optimal bw {max_normalized_bw} with distances {d}")


if __name__ == "__main__":
    main()

