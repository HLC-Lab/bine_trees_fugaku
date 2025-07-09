#!/usr/bin/env python3
import math
from itertools import product

distances = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341, 683, 1365, 2731, 5461, 10923, 21845, 43691, 87381, 174763, 349525]
# smallest_negabinaries[i] contains the smallest number that can be represented with a negabinary of i bits
smallest_negabinaries = [0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762]
# largest_negabinaries[i] contains the largest number that can be represented with a negabinary of i bits
largest_negabinaries = [0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525]

# Gets the distance of the peer of rank 'sender' at the step-th step
def get_peer_distance(sender, step, num_ranks, collective):
    sign = (-1)**step
    if sender % 2 != 0:        
        sign *= -1
    if collective == "ALLGATHER": # Distance "halving"
        return sign*distances[int(math.ceil(math.log2(num_ranks))) - step - 1]
    else: # Distance "doubling"
        return sign*distances[step]

# Gets the peer of rank 'sender' at the step-th step
def get_peer(sender, step, num_ranks, collective):
    distance = get_peer_distance(sender, step, num_ranks, collective)
    return (sender + distance) % num_ranks

# Recursive function to build the tree
# @param num_nodes: number of nodes participating in the collective
# @param root: rank of the root of the tree
# @param step: step of the tree building process
# @param collective: collective to build the tree for
# @param children: list of children for each rank
# @param parent: list of parent for each rank
def build_tree_bine_flat_inner(num_nodes, root, step, collective, children, parent):
    for s in range(step + 1, math.floor(math.log2(num_nodes))):
        peer = get_peer(root, s, num_nodes, collective)

        # Since we stop at the step before the last one, we can never
        # reach a node twice, not even if num_nodes is not a power of two
        assert(parent[peer] is None)
        
        if parent[peer] is None:
            if children[root] is None:
                children[root] = []
            children[root].append(peer)
            parent[peer] = root

            build_tree_bine_flat_inner(num_nodes, peer, s, collective, children, parent)

# Rotate a list to the right by n (in place, instead of returning a new list)
def rotate(l, n):
    l[:] = l[-n:] + l[:-n]

# Applies a DFS visit to the tree rooted in node, and assigns a label to each node
# In this way, we guarantee that subtrees are labeled contiguously
# @param children: list of children for each rank
# @param root: root of the subtree to visit
# @param relabels: list of labels for each rank
# @param next_label: is a list with a single element, ugly way to do pass by reference
def dfs(children, root, relabels, next_label):
    if children[root] is not None:
        for c in children[root]:
            dfs(children, c, relabels, next_label)
    relabels[root] = next_label[0]
    next_label[0] -= 1

def decimal_to_negabinary(decimal):
    # Special case: if the decimal is 0, return "0"
    if decimal == 0:
        return "0"
    
    negabinary = []
    while decimal != 0:
        decimal, remainder = divmod(decimal, -2)
        # Adjust for negative remainders
        if remainder < 0:
            decimal += 1
            remainder += 2
        negabinary.append(str(remainder))
    
    # Negabinary digits are collected in reverse order
    return ''.join(reversed(negabinary))

# For non-powers of 2, a rank might have two valid representations,
# one of which has a one as the MSB (indicating that the node is reached in the last step -- i.e., that it is a leaf of the binomial tree).
# I should always consider the representation that does not have a 1 as the MSB, since the only ranks
# that have a 1 as the MSB are those that can **ONLY** be reached in the last step.
def get_rank_negabinary(q, num_nodes):
    # Generates the negabinary both for q  and for  q - num_nodes (if q is odd)
    # Generates the negabinary both for -q and for -q + num_nodes (if q is even)
    num_bits = int(math.ceil(math.log2(num_nodes)))
    
    nba = None
    nbb = None    

    if q % 2:
        if smallest_negabinaries[num_bits] <= q <= largest_negabinaries[num_bits]:
            nba = decimal_to_negabinary(q).zfill(num_bits)
        if smallest_negabinaries[num_bits] <= q - num_nodes <= largest_negabinaries[num_bits]:
            nbb = decimal_to_negabinary(q - num_nodes).zfill(num_bits)
    else:
        if smallest_negabinaries[num_bits] <= -q <= largest_negabinaries[num_bits]:
            nba = decimal_to_negabinary(-q).zfill(num_bits)
        if smallest_negabinaries[num_bits] <= -q + num_nodes <= largest_negabinaries[num_bits]:
            nbb = decimal_to_negabinary(-q + num_nodes).zfill(num_bits)

    assert(nba is not None or nbb is not None)

    if nba is None and nbb is not None:
        return nbb
    elif nba is not None and nbb is None:
        return nba
    else: # Check MSB
        if nba[0] == "1":
            return nba
        else:
            return nbb

def get_rank_negabinary_old(q, num_nodes):
    # Generates the negabinary both for q  and for  q - num_nodes (if q is odd)
    # Generates the negabinary both for -q and for -q + num_nodes (if q is even)
    if q % 2:
        nba = decimal_to_negabinary(q)
        nbb = decimal_to_negabinary(q - num_nodes)
    else:
        nba = decimal_to_negabinary(-q)
        nbb = decimal_to_negabinary(-q + num_nodes)

    # Only one of those will have less (or equal) to log2(num_nodes) bits
    assert(len(nba) > math.ceil(math.log2(num_nodes)) or len(nbb) > math.ceil(math.log2(num_nodes)))
    
    nb = 0
    # Check which of the two negabinaries is the correct one
    if len(nba) <= math.ceil(math.log2(num_nodes)):
        nb = nba
    else:  
        nb = nbb

    # Expand with zeros to the left
    nb = nb.zfill(int(math.log2(num_nodes)))

    return nb

def int_to_bin(n, num_bits):
    return bin(n)[2:].zfill(num_bits)
    
def get_steps_list(q, num_nodes):
    # Get the negabinary representation of the rank
    nb = get_rank_negabinary(q, num_nodes)
    bit_indexes = int_to_bin(int(nb, 2) ^ (int(nb, 2) >> 1), int(math.log2(num_nodes)))
    # Revert the bit indexes
    bit_indexes = bit_indexes[::-1]
    # Return the indexes of the steps
    return [i for i, bit in enumerate(bit_indexes) if bit == '1']
    ''' 
    # Now scan the negabinary (backwards -- from LSB) to get the sequence of steps
    steps = []
    for i in range(len(nb)):
        pos = len(nb) - i - 1
        if nb[pos] == '1':
            if pos == 0 or nb[pos - 1] == '0': # If the previous bit is 0 or if we reached the MSB
                steps.append(i)
        if nb[pos] == '0':
            if pos > 0 and nb[pos - 1] == '1': # If the previous bit is 1 or if we reached the MSB
                steps.append(i)
    return steps
    '''

def get_remapped_rank(q, num_nodes):
    # Get the negabinary representation of the rank
    nb = get_rank_negabinary(q, num_nodes)
    bit_indexes = int_to_bin(int(nb, 2) ^ (int(nb, 2) >> 1), int(math.log2(num_nodes)))
    # Revert the bit indexes and convert back to int
    return int(bit_indexes[::-1], 2)

# Build the tree for the bine-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param root: rank of the root of the tree
# @param step: step of the tree building process
# @param collective: collective to build the tree for
# @param children: list of children for each rank
# @param parent: list of parent for each rank
# @param relabels: list of new labels for each rank
def build_tree_bine_flat(num_nodes, root, step, collective, children, parent, relabels):
    # I build the tree rooted in 0, and then I rotate it appropriately
    reference_root = 0
    assert(reference_root == 0) # We use a variable just for mnemonic purposes, but the code assumes 0
    build_tree_bine_flat_inner(num_nodes, reference_root, step, collective, children, parent)

    # If num_nodes is not a power of two, some nodes might have never been reached.
    # Add those to the tree, by considering the distance they would talk to in the last step
    for r in range(num_nodes):
        if parent[r] is None and r != reference_root:
            peer = get_peer(r, math.floor(math.log2(num_nodes)), num_nodes, collective)
            assert(parent[r] is None)
            parent[r] = peer
            if children[peer] is None:
                children[peer] = []
            children[peer].append(r)


    # OLD WAY TO RELABEL THE RANKS
    # Apply a DFS visit to the tree, to assign a label to each rank
    #dfs(children, reference_root, relabels, [num_nodes - 1])

    ########################################
    # Relabel ranks (only for powers of 2) #
    ########################################
    if not is_power_of_two(num_nodes):
        for r in range(num_nodes):
            relabels[r] = r
    else:
        # Generate the negabinary representation for each rank
        relabels[0] = 0
        for q in range(1, num_nodes):
            relabels[q] = get_remapped_rank(q, num_nodes)
    
    #print(relabels)
    assert(sorted(relabels) == list(range(num_nodes)))

    # Renumber the ranks so that the root is root
    if root % 2: # If it is an odd rank, compute the binomial tree rooted in 1 as a reference        
        children[0].remove(1)
        children[1].append(0)
        parent[0] = 1
        parent[1] = None
        gap = root - 1
    else:
        gap = root - 0

    # Increase any rank id by gap (modulo num_nodes) (children)
    for c in children:
        if c is not None:
            for i in range(len(c)):
                c[i] = (c[i] + gap) % num_nodes
    rotate(children, gap)

    # Increase any rank id by gap (modulo num_nodes) (parent)
    for i in range(len(parent)):
        if parent[i] is not None:
            parent[i] = (parent[i] + gap) % num_nodes
    rotate(parent, gap)

# Validate the schedule of the bine-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param collective: collective to build the tree for
def validate_schedule(num_nodes, collective):
    received_data = [None]*num_nodes
    for r in range(num_nodes):
        received_data[r] = [r]

    for r in range(num_nodes):
        children = [None]*num_nodes
        parent = [None]*num_nodes  
        relabels = [None]*num_nodes      
        build_tree_bine_flat(num_nodes, r, -1, collective, children, parent, relabels)        
        c = children[r]    
        while len(c):
            new_c = []
            for p in c:
                received_data[p].append(r)
                if children[p] is not None:
                    new_c.extend(children[p])
            c = new_c
    for r in range(num_nodes):
        assert(sorted(received_data[r]) == list(range(num_nodes)))
    print("Validation succeeded on {} nodes.".format(num_nodes))

# Validate the schedule of the bine-flat algorithm for several
# values of num_nodes
def validate_all():
    for n in [4, 8, 6, 10, 12, 14, 18, 30, 32, 64, 128, 256]:
        validate_schedule(n, "REDUCE-SCATTER")

# Check if a number is a power of two
# @param n: number to check
def is_power_of_two(n):
    return n != 0 and (n & (n - 1)) == 0

# Get the range of children in the subtree rooted in r
# @param children: list of children for each rank
# @param relabels: list of new labels for each rank
# @param num_nodes: number of nodes participating in the collective
# @param r: the rank for which we want to get the range
def get_children_range(children, relabels, num_nodes, r):
    # Find the minimum and maximum (remapped) rank (i.e., block index) in that subtree
    min_block_id = num_nodes
    max_block_id = -1
    # Collapse the subtree into a single list
    # Do a DFS visit of the subtree
    stack = [r]
    relabeled_children = []
    while len(stack):
        node = stack.pop()
        relabeled_children.append(relabels[node])
        if relabels[node] < min_block_id:
            min_block_id = relabels[node]
        if relabels[node] > max_block_id:
            max_block_id = relabels[node]
        if children[node] is not None:
            stack.extend(children[node])
    return min_block_id, max_block_id, relabeled_children

# Get the tx info for the bine-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param collective: collective to build the tree for
# @param root: rank of the root of the collective
# @param rank: the rank for which we want to get the tx info (for multi-root collectives, it is the same as root)
def get_tx_info_bine_flat_contiguous(num_nodes, collective, root, rank=-1):
    if collective == "ALLREDUCE" or collective == "ALLGATHER" or collective == "REDUCE-SCATTER":
        assert(is_power_of_two(num_nodes))
    if rank == -1:
        rank = root
    
    # For the BCAST, each node starts when it is reached by the data 
    # coming from the root
    gap = 0
    if (collective == "BCAST" or collective == "SCATTER"):
        # Remap the tree. I.e., the root is always 0
        if root % 2 == 0:
            gap = root
            root = 0
            rank -= gap
        else:
            gap = root - 1
            root = 1
            rank -= gap
    
    # Build the binomial tree (loopless)
    children = [None]*num_nodes
    parent = [None]*num_nodes        
    relabels = [None]*num_nodes
    build_tree_bine_flat(num_nodes, root, -1, collective, children, parent, relabels)
    
    starting_step = 0

    # For the BCAST, each node starts when it is reached by the data 
    # coming from the root
    if (collective == "BCAST" or collective == "SCATTER") and rank != root:
        # Find the step from which I should start to send data
        #steps = get_steps_list(rank, num_nodes)
        #starting_step = sorted(steps)[-1] + 1
        nb = get_rank_negabinary(rank, num_nodes)
        # Just find the most significant 1 in the negabinary representation. 
        # That's the step in which I am going to be reached by rank 0 data.
        # Since index 0 would be the most significant bit, we reverse the representation
        # so that indexes are consistent with the bit position (i.e., 0 is the rightmost)

        # +1 because I want to start from the next step (on the first step I only receive)        
        # I do the max with 0 because when a 1 is not found (e.g., for rank 0 representation),
        # the rfind returns -1, but in practice I want to start from step 0.
        starting_step = max(nb[::-1].rfind('1'), 0) + 1 

    #print(starting_step)
    # Now use the two pieces of info computed above to build the tx info
    min_block_id_r = 0
    max_block_id_r = num_nodes - 1
    num_blocks = num_nodes / 2
    for s in range(starting_step, math.ceil(math.log2(num_nodes))):
        peer = get_peer(root, s, num_nodes, collective)
        if peer in children[root]: # Just to manage the case in which I might skip a step
            # Prepare scheduling info according to collectives
            if collective == "BCAST":
                peer = get_peer(rank, s, num_nodes, collective)
                peer = (peer + gap) % num_nodes
                print("Step {}: Peer: {} Blocks: (0-{})".format(s, peer, num_nodes - 1))
            elif collective == "ALLREDUCE" or collective == "REDUCE-SCATTER" or collective == "ALLGATHER" or collective == "SCATTER":
                min_block_id_s = min_block_id_r
                max_block_id_s = max_block_id_r

                middle = (min_block_id_r + max_block_id_r + 1) // 2 # = min + (max-min)/2
                if relabels[rank] < middle:
                    min_block_id_s = middle
                    max_block_id_r = middle - 1
                else:
                    max_block_id_s = middle - 1
                    min_block_id_r = middle

                num_blocks /= 2                
                if collective == "SCATTER":
                    peer = get_peer(rank, s, num_nodes, collective)

                print("Step {}: Peer: {} Blocks to send: ({}-{})".format(s, peer, min_block_id_s, max_block_id_s))

validate_all()
get_tx_info_bine_flat_contiguous(6, "BCAST", 0, 5)