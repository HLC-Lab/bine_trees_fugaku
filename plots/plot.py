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

algo_names = {}
algo_names["default"] = "Default"
algo_names["ring"] = "Ring"
algo_names["lat_BBBN"] = "Swing (L)"
algo_names["bw_BBBN"] = "Swing (B)"
algo_names["lat_CONT"] = "Swing (L) - C"
algo_names["bw_CONT"] = "Swing (B) - C"
algo_names["recdoub_l"] = "RecDoub (L)"
algo_names["recdoub_b"] = "RecDoub (B)"


merge = True
best_algo = {}
def plot(arch, p):
    df = pd.DataFrame()
    for n in [1, 8, 64, 512, 2048, 16384, 131072, 1048576, 8388608, 67108864]:
        def_bw = 0
        #for algo in ["default", "lat_BBB", "bw_BBB", "lat_BBBN", "bw_BBBN"]:        
        best_swing_bw = 0
        best_recdoub_bw = 0
        for algo in ["default", "lat_BBBN", "bw_BBBN", "lat_CONT", "bw_CONT", "recdoub_l", "recdoub_b", "ring"]:
            k = (arch, str(p))
            if k in paths:
                vpath = paths[k]
            else:
                continue
            filename = vpath + "/" + str(p) + "_" + str(n) + "_" + algo + ".csv"
            if os.path.exists(filename):
                data_real = pd.read_csv(filename, sep=" ", skipfooter=1)                    
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
                if algo == "default":
                    def_bw = data_real["Bandwidth (Gb/s)"].mean()
                data_real["Normalized Bandwidth"] = data_real["Bandwidth (Gb/s)"]/def_bw

                if merge:
                    meanbw = data_real["Normalized Bandwidth"].mean()
                    if ("BBBN" in algo or "CONT" in algo) and meanbw > best_swing_bw:
                        data_real["Algo"] = "Swing"
                        best_swing_bw = meanbw
                        best_algo["Swing"] = data_real.copy()
                    elif ("recdoub" in algo) and meanbw > best_recdoub_bw:                        
                        data_real["Algo"] = "RecDoub"
                        best_recdoub_bw = meanbw
                        best_algo["RecDoub"] = data_real.copy()
                    elif algo == "default":
                        data_real["Algo"] = "Default"
                        best_algo["Default"] = data_real.copy()
                    elif algo == "ring":
                        data_real["Algo"] = "Ring"
                        best_algo["Ring"] = data_real.copy()
                else:
                    data_real["Algo"] = algo_names[algo]
                    df = pd.concat([df, data_real])
        if merge:
            for ba in ["Swing", "RecDoub", "Default", "Ring"]:
                if ba in best_algo:
                    df = pd.concat([df, best_algo[ba]])

    df.reset_index(drop=True, inplace=True)
    df["Time (ms)"] = df["Time (us)"] / 1000.0

    rows = 1
    cols = 1
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.boxplot(data=df, \
                     x="Size", y="Normalized Bandwidth", hue="Algo", showmeans=True, 
                     meanprops={
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, ax=axes)    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_box.pdf", format='pdf', dpi=100)
    plt.clf()


    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.lineplot(data=df, \
                      x="Size", y="Bandwidth (Gb/s)", hue="Algo", style="Algo", sort=False,
                      markers=True, dashes=True, ax=axes)
    plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_line.pdf", format='pdf', dpi=100)
    plt.clf()

ps = {}
ps["daint"] = [18, 30, 32]
ps["daint_ad3"] = [62, 64]
ps["deep-est"] = [14, 16, 30, 32]
ps["daint_sameswitch"] = [4]
ps["daint_twocabs"] = [8]
ps["daint_twocabs_ad3"] = [8]

#archs = ["daint", "daint_sameswitch", "daint_twocabs", "daint_twocabs_ad3"]
archs = ["daint_ad3"]
def main():
    # Load paths
    with open("../data/description.csv", mode='r') as infile:
        reader = csv.reader(infile)    
        global paths
        paths = {(rows[0],rows[1]):"../data/" + rows[2] for rows in reader}

    for arch in archs:
        for p in ps[arch]:
            plot(arch, p)


if __name__ == "__main__":
    main()