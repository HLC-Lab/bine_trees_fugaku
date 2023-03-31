import numpy as np
import seaborn as sns

def main():
    arch = "daint"
    timestamp = ""
    p = 10
    n = 1024
    filename = "../data/" + arch + "/" + timestamp + "/" + str(p) + "/" + str(n) + ".csv"
    if os.path.exists(filename):
        



if __name__ == "__main__":
    main()