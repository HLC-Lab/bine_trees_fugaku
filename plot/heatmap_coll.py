import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def human_readable_size(num_bytes):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num_bytes < 1024:
            return f"{int(num_bytes)} {unit}"
        num_bytes /= 1024
    return f"{int(num_bytes)} PiB"

def main():
    parser = argparse.ArgumentParser(description="Generate graphs")
    parser.add_argument("--folder", required=True, help="Path of the folder containg the data")
    parser.add_argument("--plot_type", required=False, help="Type of plot to generate", default="line")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Data folder {args.folder} not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Get the system name from the folder name
    system_name = args.folder.split("/")[1]
    timestamp = args.folder.split("/")[2]
    
    # Scan metadata line by line
    # Read file
    metadata_file = f"results/{system_name}_metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.folder + "/aggregated_results_summary.csv")

    # Drop the columns I do not need
    df = df[["buffer_size", "collective_type", "algo_name", "mean", "median"]]

    # If system name is "fugaku", drop all the algo_name starting with uppercase "RECDOUB"
    if system_name == "fugaku":
        df = df[~df["algo_name"].str.startswith("RECDOUB")]

    # Create masks
    swing_mask = df["algo_name"].str.lower().str.startswith("swing")
    non_swing_mask = ~swing_mask

    # Grouping keys
    group_keys = ["buffer_size", "collective_type"]

    # Find best swing per group
    best_swing = df[swing_mask].loc[
        df[swing_mask].groupby(group_keys)["mean"].idxmin()
    ].copy()
    best_swing["algo_name"] = "best_swing"

    # Find best non-swing per group
    best_other = df[non_swing_mask].loc[
        df[non_swing_mask].groupby(group_keys)["mean"].idxmin()
    ].copy()
    best_other["algo_name"] = "best_other"

    # Combine everything
    augmented_df = pd.concat([df, best_swing, best_other], ignore_index=True)

    # Combine back
    for m in ["mean", "median"]:
        augmented_df["bandwidth_" + m] = ((augmented_df["buffer_size"]*8.0)/(1000.0*1000*1000)) / (augmented_df[m].astype(float) / (1000.0*1000*1000))


    # Generate plot
    # Keep only best_swing and best_other
    best_df = augmented_df[augmented_df["algo_name"].isin(["best_swing", "best_other"])]

    # Pivot to get one column per algo_name
    pivot = best_df.pivot_table(
        index=["buffer_size", "collective_type"],
        columns="algo_name",
        values="bandwidth_mean"
    ).reset_index()

    # Compute ratio
    pivot["bandwidth_ratio"] = pivot["best_swing"] / pivot["best_other"]

    # Pivot for heatmap with sizes on the x-axis
    heatmap_data = pivot.pivot(
        index="collective_type",
        columns="buffer_size",
        values="bandwidth_ratio"
    )
    
    # Pivot again for heatmap (collective_type as rows, buffer_size as columns)
    heatmap_data = pivot.pivot(
        index="collective_type",
        columns="buffer_size",
        values="bandwidth_ratio"
    )

    custom_row_order = [
        "ALLREDUCE",
        "BCAST",
        "REDUCE",
        "ALLGATHER",
        "ALLTOALL",
        "GATHER",
        "REDUCE_SCATTER",
        "SCATTER",
    ]
    heatmap_data = heatmap_data.reindex(custom_row_order)    

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        cbar_kws={"label": "Bandwidth Ratio (best_swing / best_other)"},
        mask=heatmap_data.isna()
    )
    # For each column use the corresponding buffer_size_hr rather than buffer_size as labels
    # Get all the column names, sort them (numerically), and the apply to each of them the human_readable_size function
    # to get the human-readable size
    # Then set the x-ticks labels to these human-readable sizes
    # Get heatmap_data.columns and convert to a list of int
    buffer_sizes = heatmap_data.columns.astype(int).tolist()
    buffer_sizes.sort()
    buffer_sizes = [human_readable_size(int(x)) for x in buffer_sizes]
    # Use buffer_sizes as labels
    plt.xticks(ticks=np.arange(len(buffer_sizes)) + 0.5, labels=buffer_sizes, rotation=45)

    plt.title("Bandwidth Ratio: best_swing vs best_other")
    plt.xlabel("Vector Size")
    plt.ylabel('')
    plt.tight_layout()

    # Save as PDF
    plt.savefig("plot/bandwidth_ratio_heatmap.pdf", bbox_inches="tight")




if __name__ == "__main__":
    main()

