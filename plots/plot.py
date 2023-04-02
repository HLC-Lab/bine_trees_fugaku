import numpy as np
import seaborn as sns
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt

paths = {}

sizes = {}

sizes[1] = "4B"
sizes[8] = "32B"
sizes[64] = "256B"
sizes[512] = "2KiB"
sizes[2048] = "16KiB"
sizes[16384] = "64KiB"
sizes[131072] = "512KiB"
sizes[1048576] = "4MiB" 
sizes[8388608] = "32MiB"
sizes[67108864] = "256MiB"

def main():
    # Load paths
    with open("../data/description.csv", mode='r') as infile:
        reader = csv.reader(infile)    
        global paths
        paths = {(rows[0],rows[1]):"../data/" + rows[2] for rows in reader}

    arch = "daint"
    p = 30

    df = pd.DataFrame()
    for n in [1, 8, 64, 512, 2048, 16384, 131072, 1048576, 8388608, 67108864]:
        def_bw = 0
        #for algo in ["default", "lat_BBB", "bw_BBB", "lat_BBBN", "bw_BBBN"]:        
        for algo in ["default", "lat_BBBN", "bw_BBBN"]:
            k = (arch,str(p))
            if k in paths:
                vpath = paths[k]
            filename = vpath + "/" + str(p) + "_" + str(n) + "_" + algo + ".csv"
            if os.path.exists(filename):
                data_real = pd.read_csv(filename, sep=" ")                    
                if len(data_real) == 0:
                    continue
                data_real = data_real.loc[:, ~data_real.columns.str.contains('^Unnamed')]
                colnames_ranks = []
                for r in range(p):
                    colnames_ranks += ["Rank" + str(r) + "Time(us)"]
                data_real["Time (us)"] = data_real[colnames_ranks].max(axis=1)
                data_real["System"] = arch
                data_real["Nodes"] = p
                data_real["Size (B)"] = n*4
                data_real["Size"] = sizes[n]
                data_real["Bandwidth (Gb/s)"] = 2*((data_real["Size (B)"]*8) / (data_real["Time (us)"]*1000.0)).astype(float)
                data_real["Algo"] = algo
                if algo == "default":
                    def_bw = data_real["Bandwidth (Gb/s)"].mean()
                data_real["Normalized Bandwidth"] = data_real["Bandwidth (Gb/s)"]/def_bw
                df = pd.concat([df, data_real])
    df.reset_index(drop=True, inplace=True)
    df["Time (ms)"] = df["Time (us)"] / 1000.0

    rows = 1
    cols = 1
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)

    '''    
    ax = sns.lineplot(data=df, \
                      x="Size", y="Bandwidth (Gb/s)", hue="Algo", style="Algo", sort=False,
                      markers=True, dashes=True, ax=axes)
    plt.xscale('log')
    '''
    ax = sns.boxplot(data=df, \
                     x="Size", y="Normalized Bandwidth", hue="Algo", showmeans=True, 
                     meanprops={
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, ax=axes)

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + ".pdf", format='pdf', dpi=100)
    plt.clf()

if __name__ == "__main__":
    main()