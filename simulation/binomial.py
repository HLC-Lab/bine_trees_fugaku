#!/usr/bin/env python3

import math

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
# @param reached_at_step: step at which each rank has been reached
def build_tree_swing_flat_inner(num_nodes, root, step, collective, children, parent, reached_at_step):
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
            reached_at_step[peer] = s

            build_tree_swing_flat_inner(num_nodes, peer, s, collective, children, parent, reached_at_step)

# Rotate a list to the right by n (in place, instead of returning a new list)
def rotate(l, n):
    l[:] = l[-n:] + l[:-n]

# Build the tree for the swing-flat algorithm
# @param num_nodes: number of nodes participating in the collective
# @param root: rank of the root of the tree
# @param step: step of the tree building process
# @param collective: collective to build the tree for
# @param children: list of children for each rank
# @param parent: list of parent for each rank
# @param reached_at_step: step at which each rank has been reached
def build_tree_swing_flat(num_nodes, root, step, collective, children, parent, reached_at_step):
    # I build the tree rooted in 0, and then I rotate it appropriately
    build_tree_swing_flat_inner(num_nodes, 0, step, collective, children, parent, reached_at_step)

    # If num_nodes is not a power of two, some nodes might have never been reached.
    # Add those to the tree, by considering the distance they would talk to in the last step
    for r in range(num_nodes):
        if reached_at_step[r] is None and r != 0:
            peer = get_peer(r, math.floor(math.log2(num_nodes)), num_nodes, collective)
            assert(parent[r] is None)
            parent[r] = peer
            if children[peer] is None:
                children[peer] = []
            children[peer].append(r)
            reached_at_step[r] = math.floor(math.log2(num_nodes))

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
def validate_schedule(num_nodes, collective):
    received_data = [None]*num_nodes
    for r in range(num_nodes):
        received_data[r] = [r]

    for r in range(num_nodes):
        children = [None]*num_nodes
        parent = [None]*num_nodes        
        reached_at_step = [None]*num_nodes
        build_tree_swing_flat(num_nodes, r, -1, collective, children, parent, reached_at_step)        
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
    for n in [4, 8, 6, 10, 12, 14, 18]:
        validate_schedule(n, "REDUCE-SCATTER")

# next_label is a list with a single element, ugly way to do pass by reference
def dfs(children, node, relabels, next_label):
    if children[node] is not None:
        for c in children[node]:
            dfs(children, c, relabels, next_label)
    relabels[node] = next_label[0]
    next_label[0] -= 1

def get_tx_info_swing_flat(num_nodes, collective, r):
    # Build the binomial tree (loopless)
    children = [None]*num_nodes
    parent = [None]*num_nodes        
    reached_at_step = [None]*num_nodes
    build_tree_swing_flat(num_nodes, r, -1, collective, children, parent, reached_at_step)
    
    # Build the binomial tree of a reference rank, to use to renumber the ranks
    reference_root_for_renumbering = 0
    children_renum = [None]*num_nodes
    parent_renum = [None]*num_nodes        
    reached_at_step_renum = [None]*num_nodes
    build_tree_swing_flat(num_nodes, reference_root_for_renumbering, reference_root_for_renumbering, -1, collective, children_renum, parent_renum, reached_at_step_renum)
    relabels = [None]*num_nodes
    dfs(children_renum, reference_root_for_renumbering, relabels, [num_nodes - 1])

    # Now use the two pieces of info computed above to build the tx info
    print(relabels)
    print(children)
    for s in range(math.ceil(math.log2(num_nodes))):
        peer = get_peer(r, s, num_nodes, collective)
        if peer in children[r]: # Just to manage the case in which I might skip a step
            # Find the minimum and maximum (remapped) rank (i.e., block index) in that subtree
            min_block_id = num_nodes
            max_block_id = -1
            # Collapse the subtree into a single list
            # Do a DFS visit of the subtree
            stack = [peer]
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
            print("Step {}: ({}-{}) {}".format(s, min_block_id, max_block_id, sorted(relabeled_children)))

validate_all()
get_tx_info_swing_flat(14, "REDUCE-SCATTER", 2)