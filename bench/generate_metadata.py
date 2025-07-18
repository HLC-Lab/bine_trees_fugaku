import os
import csv
import sys
from typing import Optional

RESULTS_DIR = "../results/"

# Update or create metadata CSV
def update_metadata(system_name: str, timestamp: str, test_id: int, number_of_nodes: int,
                    collective_type: str, mpi_lib: str, mpi_lib_version : str,
                    libbine_version: str, cuda: str, output_level: str, mpi_op: Optional[str], notes: Optional[str]):
    output_file = os.path.join(RESULTS_DIR, f"{system_name}_metadata.csv")

    # Check if file exists to determine whether to write the header
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "test_id",
            "nnodes",
            "collective_type",
            "mpi_lib",
            "mpi_lib_version",
            "libbine_version",
            "CUDA",
            "output_level",
            "MPI_Op",
            "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the new metadata row
        writer.writerow({
            "timestamp": timestamp,
            "test_id": test_id,
            "nnodes": number_of_nodes,
            "tasks_per_node": 1,
            "collective_type": collective_type,
            "mpi_lib": mpi_lib,
            "mpi_lib_version": mpi_lib_version,
            "libbine_version": libbine_version,
            "gpu_awareness": "no",
            "gpu_lib": "null",
            "gpu_lib_version": "null",
            "output_level": output_level,
            "MPI_Op": mpi_op,
            "notes": notes
        })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {__file__} <test_id>", file=sys.stderr)
        sys.exit(1)

    test_id = sys.argv[1]
    if not test_id.isdigit():
        print(f"{__file__}: test_id must be an integer.", file=sys.stderr)
    test_id = int(test_id)

    system_name = os.getenv('LOCATION')
    number_of_nodes = os.getenv('N_NODES')
    timestamp = os.getenv('TIMESTAMP')
    collective_type = os.getenv('COLLECTIVE_TYPE')
    mpi_lib = os.getenv('MPI_LIB')
    mpi_lib_version = os.getenv('MPI_LIB_VERSION')
    libbine_version = os.getenv('LIBBINE_VERSION')
    cuda = os.getenv('CUDA')
    output_level = os.getenv('OUTPUT_LEVEL')
    mpi_op = os.getenv('MPI_OP')
    notes = os.getenv('NOTES') or "null"
    if not (system_name and timestamp and number_of_nodes and number_of_nodes and collective_type and mpi_lib and mpi_lib_version and libbine_version and cuda and output_level):
        print (f"{__file__}: Environment variables not set.", file=sys.stderr)
        print (f"LOCATION={system_name}\nTIMESTAMP={timestamp}\nN_NODES={number_of_nodes}\nCOLLECTIVE_TYPE={collective_type}\nMPI_LIB={mpi_lib}\nMPI_LIB_VERSION={mpi_lib_version}\nLIBBINE_VERSION={libbine_version}\nCUDA={cuda}", file=sys.stderr)
        sys.exit(1)

    update_metadata(system_name, timestamp, test_id, number_of_nodes, \
                    collective_type, mpi_lib, mpi_lib_version, \
                    libbine_version, cuda, output_level, mpi_op = mpi_op, notes = notes)
    print(f"Metadata updated for {system_name} at {timestamp}.")
