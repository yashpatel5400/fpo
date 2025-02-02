import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

import utils

def main(results_dir):
    results = {
        "Truth": [],
        "Nominal": [],
        "Robust": [],
    }
    for fn in os.listdir(results_dir):
        df = pd.read_csv(os.path.join(results_dir, fn), index_col=0)
        for experiment in df:
            results[experiment].append(df[experiment])

    normalized = {}
    for method in results:
        if method == "Truth":
            continue
        normalized[method] = (np.array(results["Truth"]) - np.array(results[method])) / np.array(results["Truth"])
        print(f'{method} -- {np.mean(normalized[method])} ({np.std(normalized[method])})')

    # Perform the t-test
    _, p_value = ttest_rel(normalized["Robust"], normalized["Nominal"], alternative='less')
    print(f"p-value: {p_value}")
    print(np.sum(normalized["Robust"] < normalized["Nominal"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    args = parser.parse_args()
    main(utils.RESULTS_DIR(args.pde))