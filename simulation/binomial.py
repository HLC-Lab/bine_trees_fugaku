#!/usr/bin/env python3
import math
from itertools import product

distances = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341]

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
def build_tree_swing_flat_inner(num_nodes, root, step, collective, children, parent):
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

            build_tree_swing_flat_inner(num_nodes, peer, s, collective, children, parent)

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

# Build the tree for the swing-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param root: rank of the root of the tree
# @param step: step of the tree building process
# @param collective: collective to build the tree for
# @param children: list of children for each rank
# @param parent: list of parent for each rank
# @param relabels: list of new labels for each rank
def build_tree_swing_flat(num_nodes, root, step, collective, children, parent, relabels):
    # I build the tree rooted in 0, and then I rotate it appropriately
    reference_root = 0
    assert(reference_root == 0) # We use a variable just for mnemonic purposes, but the code assumes 0
    build_tree_swing_flat_inner(num_nodes, reference_root, step, collective, children, parent)

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
            
            # Now scan the negabinary (backwards -- from LSB) to get the sequence of steps
            steps = []
            for i in range(len(nb)):
                pos = len(nb) - i - 1
                if nb[pos] == '1':
                    if pos == 0 or nb[pos - 1] == '0': # If the previous bit is 0 or if we reached the MSB
                        steps.append(pos)
                if nb[pos] == '0':
                    if pos > 0 and nb[pos - 1] == '1': # If the previous bit is 1 or if we reached the MSB
                        steps.append(pos)
            
            binary = [0]*int(math.log2(num_nodes))
            for s in steps:
                binary[s] = 1
            
            # Convert binary to decimal
            decimal = 0
            for i in range(len(binary)):
                decimal += binary[i]*2**i
            
            #print("Remapping rank {} to {} (nb: {}) (steps: {})".format(q, decimal, nb, steps))
            relabels[q] = decimal
    
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

# Validate the schedule of the swing-flat algorithm
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
        build_tree_swing_flat(num_nodes, r, -1, collective, children, parent, relabels)        
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

# Validate the schedule of the swing-flat algorithm for several
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

# Get the tx info for the swing-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param collective: collective to build the tree for
# @param root: rank of the root of the collective
# @param rank: the rank for which we want to get the tx info (for multi-root collectives, it is the same as root)
def get_tx_info_swing_flat_contiguous(num_nodes, collective, root, rank=-1):
    assert(is_power_of_two(num_nodes))
    if rank == -1:
        rank = root
    # Build the binomial tree (loopless)
    children = [None]*num_nodes
    parent = [None]*num_nodes        
    relabels = [None]*num_nodes
    build_tree_swing_flat(num_nodes, root, -1, collective, children, parent, relabels)
    
    starting_step = 0

    # For the BCAST, each node starts when it is reached by the data 
    # coming from the root
    if (collective == "BCAST" or collective == "SCATTER") and rank != root:
        stack = children[root]
        while len(stack):
            if rank in stack:
                break
            new_stack = []
            for c in stack:
                if children[c] is not None:
                    new_stack.extend(children[c])
            stack = new_stack
            starting_step += 1
        starting_step += 1 # In the first step I only receive

    #print(starting_step)
    # Now use the two pieces of info computed above to build the tx info
    for s in range(starting_step, math.ceil(math.log2(num_nodes))):
        peer = get_peer(root, s, num_nodes, collective)
        if peer in children[root]: # Just to manage the case in which I might skip a step
            min_block_id = 0
            max_block_id = 0 
            # Prepare scheduling info according to collectives
            if collective == "BCAST":
                peer = get_peer(rank, s, num_nodes, collective)
                min_block_id = 0
                max_block_id = num_nodes - 1
            elif collective == "ALLREDUCE" or collective == "REDUCE-SCATTER" or collective == "ALLGATHER" or collective == "SCATTER":
                min_block_id, max_block_id, relabeled_children = get_children_range(children, relabels, num_nodes, peer)
                
                # Check that the range is actually contiguous and without holes
                for c in relabeled_children:
                    assert(c >= min_block_id and c <= max_block_id)
                
                if collective == "SCATTER":
                    peer = get_peer(rank, s, num_nodes, collective)

            #print("Step {}: Peer: {} Blocks: ({}-{})".format(s, peer, min_block_id, max_block_id))

validate_all()
get_tx_info_swing_flat_contiguous(8, "ALLREDUCE", 0)