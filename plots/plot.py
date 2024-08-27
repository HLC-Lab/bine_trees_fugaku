import numpy as np
import seaborn as sns
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams


matplotlib.rc('pdf', fonttype=42) # To avoid issues with camera-ready submission
sns.set_style("whitegrid")
#sns.set_context("paper")
rcParams['figure.figsize'] = 8,4.5

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
algo_names["lat_old_CONT"] = "Swing Old (L) - C"
algo_names["bw_old_CONT"] = "Swing Old (B) - C"
algo_names["lat"] = "Swing (L)"
algo_names["bw"] = "Swing (B)"
algo_names["bw_cont"] = "Swing (B - CONT)"
algo_names["bw_coalesce"] = "Swing (B - COAL)"
algo_names["recdoub_l"] = "RecDoub (L)"
algo_names["recdoub_b"] = "RecDoub (B)"
algo_names["default_1"] = "Default (1)"
algo_names["default_2"] = "Default (2)"
algo_names["default_3"] = "Default (3)"
algo_names["default_4"] = "Default (4)"
algo_names["default_5"] = "Default (5)"
algo_names["default_6"] = "Default (6)"


# CONF
merge = False
add_default = True

#algos_sota = ["recdoub_l", "recdoub_b", "ring", "lat_old_CONT", "bw_old_CONT"]
algos_sota = ["recdoub_l", "recdoub_b", "ring", "default_1", "default_2", "default_3", "default_4", "default_5", "default_6"]
#algos_sota = ["recdoub_l", "recdoub_b", "default"]
if add_default:
    algos_sota = ["default"] + algos_sota
#algos_swing = ["bw_BBBN", "lat_old_CONT", "bw_old_CONT"]
algos_swing = ["lat", "bw", "bw_cont", "bw_coalesce"]
algos = algos_sota + algos_swing # ATTENTION! SWING MUST ALWAYS COME AFTER SOTA FOR THE SCRIPT TO WORK CORRECTLY

best_algo = {}
def plot(arch, p):
    df = pd.DataFrame()
    df_impr_to_sota = pd.DataFrame()
    for n in [1, 8, 64, 512, 2048, 16384, 131072, 1048576, 8388608, 67108864]:
        def_bw = 0
        best_swing_bw = 0
        best_recdoub_bw = 0
        best_sota_bw = 0
        for algo in algos:
            if ("bw" in algo) and n < p:
                continue

            k = (arch, str(p))
            if k in paths:
                vpath = paths[k]
            else:
                print("No data found for " + str(k))
                continue
            filename = vpath + "/" + str(p) + "_" + str(n) + "_" + algo + ".csv"
            print("Accessing data on " + filename)
            if os.path.exists(filename):
                try:
                    data_real = pd.read_csv(filename, sep=" ", on_bad_lines='skip', comment="[") # TODO Is not actually a comment but on Leonardo there are some warning lines starting with "["
                    data_real.drop(data_real.tail(1).index, inplace=True) 
                except:
                    continue
                if len(data_real) == 0:                    
                    continue
                data_real = data_real.loc[:, ~data_real.columns.str.contains('^Unnamed')]
                colnames_ranks = []
                for r in range(p):
                    colnames_ranks += ["Rank" + str(r) + "Time(us)"]
                data_real = data_real.astype(float)
                data_real["Time (us)"] = data_real[colnames_ranks].max(axis=1)
                data_real["System"] = arch
                data_real["Nodes"] = p
                data_real["Size (B)"] = n
                data_real["Size"] = sizes[n]
                data_real["Bandwidth (Gb/s)"] = 2*((data_real["Size (B)"]*8) / (data_real["Time (us)"]*1000.0)).astype(float)                
                if algo == "default":
                    def_bw = data_real["Bandwidth (Gb/s)"].mean()
                data_real["Normalized Bandwidth"] = data_real["Bandwidth (Gb/s)"]/def_bw

                bw_mean = data_real["Bandwidth (Gb/s)"].mean()
                if algo in algos_sota and bw_mean > best_sota_bw:
                    best_sota_bw = bw_mean
                if not merge and algo in algos_swing:
                    df_tmp = pd.DataFrame()
                    name = algo_names[algo]            
                    df_tmp["Improvement (%)"] = ((data_real["Bandwidth (Gb/s)"] - best_sota_bw)/best_sota_bw)*100.0
                    df_tmp["Size"] = sizes[n]
                    df_tmp["Algorithm"] = name
                    df_impr_to_sota = pd.concat([df_impr_to_sota, df_tmp])                

                if merge:
                    meanbw = data_real["Bandwidth (Gb/s)"].mean()
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
                    if ba == "Swing":
                        # For impr plot
                        df_tmp = pd.DataFrame()
                        df_tmp["Improvement (%)"] = ((best_algo["Swing"]["Bandwidth (Gb/s)"] - best_sota_bw)/best_sota_bw)*100.0
                        df_tmp["Size"] = sizes[n]
                        df_tmp["Algorithm"] = "Swing"
                        df_impr_to_sota = pd.concat([df_impr_to_sota, df_tmp])         

                    if not add_default: # If I didn't normalize bw wrt default, now I normalize wrt Swing
                        best_algo[ba]["Normalized Bandwidth"] = best_algo[ba]["Bandwidth (Gb/s)"] / best_algo["Swing"]["Bandwidth (Gb/s)"].mean()
                    df = pd.concat([df, best_algo[ba]])

    if len(df) == 0:
        return
    df_impr_to_sota.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Time (ms)"] = df["Time (us)"] / 1000.0

    rows = 1
    cols = 1

    # Improve to sota, lines
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.pointplot(data=df_impr_to_sota, \
                      x="Size", y="Improvement (%)", hue="Algorithm", ax=axes)
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_impr_sota.pdf", format='pdf', dpi=100)
    plt.clf()

    # All, boxes
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.boxplot(data=df, \
                     x="Size", y="Normalized Bandwidth", hue="Algo", showmeans=True, 
                     showfliers=False,
                     meanprops={
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, ax=axes)    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_box.pdf", format='pdf', dpi=100)
    plt.clf()


    # All, lines
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.pointplot(data=df, \
                      x="Size", y="Bandwidth (Gb/s)", hue="Algo", ax=axes)
    #plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_line_bw.pdf", format='pdf', dpi=100)
    plt.clf()

    # All, lines
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.pointplot(data=df, \
                      x="Size", y="Time (us)", hue="Algo", ax=axes)
    #plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_line_time.pdf", format='pdf', dpi=100)
    plt.clf()

    # All, lines
    fig, axes = plt.subplots(rows, cols, figsize=(10,10), sharex=False, sharey=False)
    ax = sns.lineplot(data=df, \
                      x="Size", y="Normalized Bandwidth", hue="Algo", style="Algo", ax=axes, markers=True)
    #plt.xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("out/" + arch + "_" + str(p) + "_line_normbw.pdf", format='pdf', dpi=100)
    plt.clf()

    print("out/" + arch + "_" + str(p) + "_ plotted.")

ps = {}
ps["alps"] = [16, 32, 62, 64]
ps["daint_ad3"] = [16, 62, 64]
ps["deep-est"] = [16, 32]

ps["daint"] = [18, 30, 32]
ps["daint_sameswitch"] = [4]
ps["daint_twocabs"] = [8]
ps["daint_twocabs_ad3"] = [8]
ps["leonardo_UCXIBSL1"] = [8]
ps["lumi"] = [14, 16]
ps["leonardo_ONENIC"] = [6, 8, 14]
ps["fugaku"] = [64]

#archs = ["daint_ad3", "deep-est", "alps", "daint", "daint_sameswitch", "daint_twocabs", "daint_twocabs_ad3", "leonardo"]
archs = ["fugaku"]
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
