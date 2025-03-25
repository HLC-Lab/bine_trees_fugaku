#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
from math import log2, ceil
import argparse
import numpy as np

MIN_GROUP_SIZE = 1

def get_bytes_sent_at_step(step, vector_size, algo, num_nodes):
    if algo == "reduce-scatter":
        return vector_size / (2.0**(step+1))
    elif algo == "allgather":
        return (vector_size / num_nodes) * (2.0**(step))
    else:
        exit("Unknown algorithm")
    
def get_outgoing_bytes_step(step, start_group, end_group, vector_size, algo, family, num_nodes):
    outgoing_bytes_step = 0
    if family == "swing":
        distances_swing = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341, 683, 1365, 2731, 5461, 10923, 21845, 43691, 87381, 174763, 349525]        
        real_step = step
        if algo == "allgather":
            real_step = ceil(log2(num_nodes)) - step - 1
        for node in range(start_group, end_group + 1):
            # Compute the outgoing bytes for Swing
            peer = node
            if (step % 2 == 0 and node % 2 == 0) or (step % 2 != 0 and node % 2 != 0):
                peer = (node + distances_swing[real_step]) % num_nodes                
            else:
                peer = (node - distances_swing[real_step]) % num_nodes
            if peer < start_group or peer > end_group:
                outgoing_bytes_step += get_bytes_sent_at_step(step, vector_size, algo, num_nodes)
        return outgoing_bytes_step
    elif family == "recdoub":
        for node in range(start_group, end_group + 1):
            # Compute the outgoing bytes for Recdoub
            if algo == "reduce-scatter":
                peer = node ^ (1 << (ceil(log2(num_nodes)) - step - 1))            
            else:
                peer = node ^ (1 << step)
            if peer < start_group or peer > end_group:
                outgoing_bytes_step += get_bytes_sent_at_step(step, vector_size, algo, num_nodes)
        return outgoing_bytes_step
    elif family == "sparbit":
        for node in range(start_group, end_group + 1):
            # Start from distant nodes
            if algo == "allgather":
                peer = node ^ (1 << (ceil(log2(num_nodes)) - step - 1))            
            else:
                exit("Unknown algorithm for sparbit")
            if peer < start_group or peer > end_group:
                outgoing_bytes_step += get_bytes_sent_at_step(step, vector_size, algo, num_nodes)
        return outgoing_bytes_step        
    elif family == "bruck":
        for node in range(start_group, end_group + 1):
            # Start from distant nodes
            if algo == "allgather":
                peer = (node + 2**step) % num_nodes
            else:
                exit("Unknown algorithm for sparbit")
            if peer < start_group or peer > end_group:
                outgoing_bytes_step += get_bytes_sent_at_step(step, vector_size, algo, num_nodes)
        return outgoing_bytes_step            
    else:
        exit("Unknown family " + family)
    
    return outgoing_bytes_step


# define main function
def main():
    # Read collective from command line. Use arguments reader
    parser = argparse.ArgumentParser(description="Plots outgoing bytes distribution")
    parser.add_argument("--algo", type=str, default="reduce-scatter", help="Collective algorithm")
    parser.add_argument("--vector_size", type=int, default=1, help="Size of the vector")
    parser.add_argument("--num_nodes", type=int, default=64, help="Number of nodes")
    args = parser.parse_args()
    

    outgoing_bytes = {}
    dist_om = {}

    families = []
    if args.algo == "reduce-scatter":
        families = ["swing", "recdoub", "sparbit"]
    elif args.algo == "allgather":
        families = ["swing", "recdoub", "bruck", "sparbit"]
    else:
        exit("Unknown algorithm " + args.algo)
    
    for family in families:
        outgoing_bytes[family] = 0
        dist_om[family] = []

    for start_group in range(0, args.num_nodes):
        for end_group in range(start_group, args.num_nodes):
            group_size = end_group - start_group + 1
            if group_size < MIN_GROUP_SIZE:
                continue
            # For this specific fully connected group, compute the number
            # of outgoing bytes at each step
            for step in range(0, ceil(log2(args.num_nodes))):
                for family in families:
                    outgoing_bytes[family] += get_outgoing_bytes_step(step, start_group, end_group, args.vector_size, args.algo, family, args.num_nodes)

            for family in families:
                dist_om[family].append(outgoing_bytes[family])

    #print("Worst group (recdoub) is from", worst_group_start, "to", worst_group_end, "with", worst_group_out_rd, "outgoing bytes (vs. Swing with", worst_group_out_sw, "outgoing bytes)")

    # Plot kdplot of dist_om_swing vs. dist_om_recdoub with Seaborn (save to file)    
    for family in families:
        sns.kdeplot(dist_om[family], label=family, cut=0)
    plt.xlabel("Outgoing bytes")
    plt.ylabel("Density")
    #plt.title("Swing vs. Recdoub")
    plt.legend()
    plt.savefig(args.algo + "_" + str(args.num_nodes) + "_" + str(args.vector_size) + ".png")
    plt.clf()

    # Plot also CDF using seaborn
    for family in families:
        sns.kdeplot(dist_om[family], label=family, cumulative=True)
    plt.xlabel("Outgoing bytes")
    plt.ylabel("Density")
    # Set log x-scale
    plt.xscale("log")
    #plt.title("Swing vs. Recdoub")
    plt.legend()
    plt.savefig(args.algo + "_cdf_" + str(args.num_nodes) + "_" + str(args.vector_size) + ".png")
    plt.clf()

    # Plot the percentage reduction of swing over all the other algorithms
    for family in families:
        if family == "swing":
            continue
        percentage_reduction = np.array(dist_om[family]) / np.array(dist_om["swing"])
        sns.kdeplot(percentage_reduction, cut=0, label=family)
    plt.xlabel("Percentage reduction of Swing over other algos")
    plt.ylabel("Density")
    # Set legend
    plt.legend()
    #plt.title("Swing vs. Recdoub")
    plt.savefig(args.algo + "_reduction_" + str(args.num_nodes) + "_" + str(args.vector_size) + ".png")
    plt.clf()

    # and also CDF
    for family in families:
        if family == "swing":
            continue
        percentage_reduction = np.array(dist_om[family]) / np.array(dist_om["swing"])
        sns.kdeplot(percentage_reduction, cumulative=True, label=family)
    plt.xlabel("Percentage reduction of Swing over other algos")
    plt.ylabel("Density")
    # Set legend
    plt.legend()
    #plt.title("Swing vs. Recdoub")
    plt.savefig(args.algo + "_reduction_cdf_" + str(args.num_nodes) + "_" + str(args.vector_size) + ".png")
    plt.clf()

if __name__ == "__main__":
    main()