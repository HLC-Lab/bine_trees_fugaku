#!/usr/bin/env python3
from math import log2, ceil

distances_swing = [1, 1, 3, 5, 11, 21, 43, 85, 171, 341, 683, 1365, 2731, 5461, 10923, 21845, 43691, 87381, 174763, 349525]

NUM_NODES = 256
GROUP_OUTGOING_LINKS = 1
MIN_GROUP_SIZE = 1
SCALE_MSG_SIZE = True

dist_om_swing = []
dist_om_recdoub = []

for start_group in range(0, NUM_NODES):
    for end_group in range(start_group, NUM_NODES):
        if end_group - start_group + 1 < MIN_GROUP_SIZE:
            continue
        # For this specific fully connected group, compute the number
        # of outgoing messages at each step
        outgoing_messages_swing = 0
        outgoing_messages_recdoub = 0
        for step in range(0, ceil(log2(NUM_NODES))):
            outgoing_messages_swing_step = 0
            outgoing_messages_recdoub_step = 0
            for node in range(start_group, end_group + 1):
                # Compute the outgoing messages for Swing
                peer = node
                if (step % 2 == 0 and node % 2 == 0) or (step % 2 != 0 and node % 2 != 0):
                    peer = (node + distances_swing[step]) % NUM_NODES                
                else:
                    peer = (node - distances_swing[step]) % NUM_NODES
                if peer < start_group or peer > end_group:
                    outgoing_messages_swing_step += 1

                # Compute the outgoing messages for Recdoub
                if (node ^ (1 << step)) < start_group or (node ^ (1 << step)) > end_group:
                    outgoing_messages_recdoub_step += 1
        
            if SCALE_MSG_SIZE:
                outgoing_messages_swing_step *= (1/2)**(step+1)
                outgoing_messages_recdoub_step *= (1/2)**(step+1)

            outgoing_messages_swing += outgoing_messages_swing_step
            outgoing_messages_recdoub += outgoing_messages_recdoub_step

        #outgoing_messages_swing /= GROUP_OUTGOING_LINKS
        #outgoing_messages_recdoub /= GROUP_OUTGOING_LINKS

        dist_om_swing.append(outgoing_messages_swing)
        dist_om_recdoub.append(outgoing_messages_recdoub)

# Plot kdplot of dist_om_swing vs. dist_om_recdoub with Seaborn (save to file)
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(dist_om_swing, label="Swing", cut=0)
sns.kdeplot(dist_om_recdoub, label="Recdoub", cut=0)
plt.xlabel("Outgoing messages")
plt.ylabel("Density")
plt.title("Swing vs. Recdoub")
plt.legend()
plt.savefig("recdoub_vs_swing.png")
plt.clf()

# Plot also CDF using seaborn
sns.kdeplot(dist_om_swing, label="Swing", cumulative=True)
sns.kdeplot(dist_om_recdoub, label="Recdoub", cumulative=True)
plt.xlabel("Outgoing messages")
plt.ylabel("Density")
plt.title("Swing vs. Recdoub")
plt.legend()
plt.savefig("recdoub_vs_swing_cdf.png")