import sys

sys.path.append('.')  # noqa

from tabulate import tabulate
import argparse
import pandas as pd
import os
import numpy as np
from utils import RESULTS_DIR, PERF_MEASURE_PER_OUTPUT_TYPE, DATASETS, FAIRNESS_NAMES

DATA_NAME = "ACSPublicCoverage"
NOTION = "dem_par"
TYPE = "hard"

K_TYPE_DICT = {
    "soft": {
        "dem_par": {
            "binary": [0.01, 0.03, 0.05],
            "parallel": [0.08, 0.12, 0.16],
            "intersectional": [0.2, 0.35, 0.5]},
        "eq_opp": {
            "binary": [0.005, 0.01, 0.02],
            "parallel": [0.1, 0.15, 0.2],
            "intersectional": [0.1, 0.2, 0.3]}
    },
    "hard": {
        "dem_par": {
            "binary": [0.01, 0.05, 0.1],
            "parallel": [0.1, 0.3, 0.5],
            "intersectional": [0.5, 0.75, 1]}
    }
}

SENS_ATTR_NAME = {"binary": "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\parbox[c]{1cm}{Binary}}}",
                  "intersectional": "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\parbox[c]{1cm}{Inters.}}}",
                  "parallel": "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\parbox[c]{1cm}{Parallel}}}"}

NUM_COLS = ['_runtime', 'preprocess_time', 'postprocess_time', 'test/soft/parallel/forp',
            'test/hard/parallel/dem_par', 'test/hard/parallel/pred_eq',
            'test/hard/binary/f1_score_eq', 'test/hard/intersectional/acc_eq',
            'test/soft/intersectional/pred_eq',
            'test/hard/parallel/eq_opp', 'test/soft/parallel/eq_opp',
            'test/soft/binary/f1_score_eq', 'test/soft/intersectional/dem_par',
            'test/hard/parallel/acc_eq', 'test/hard/binary/acc_eq',
            'test/soft/parallel/acc_eq', 'test/soft/binary/eq_opp',
            'test/hard/binary/pred_par', 'test/soft/parallel/pred_par',
            'test/hard/intersectional/pred_par', 'test/f1_score',
            'test/hard/parallel/forp', 'test/hard/binary/dem_par',
            'test/hard/binary/pred_eq', 'test/soft/binary/pred_par',
            'test/hard/intersectional/eq_opp', 'test/hard/intersectional/dem_par',
            'test/soft/intersectional/f1_score_eq', 'test/auroc',
            'test/hard/binary/forp', 'test/soft/binary/forp',
            'test/soft/parallel/dem_par', 'test/soft/parallel/pred_eq',
            'test/soft/binary/dem_par', 'test/soft/binary/pred_eq',
            'test/hard/intersectional/forp', 'test/soft/intersectional/forp',
            'test/soft/intersectional/eq_opp', 'test/soft/intersectional/pred_par',
            'test/accuracy', 'training_time', 'test/hard/binary/eq_opp',
            'test/soft/binary/acc_eq', 'test/hard/intersectional/pred_eq',
            'test/hard/intersectional/f1_score_eq', 'test/hard/parallel/pred_par',
            'test/hard/parallel/f1_score_eq', 'test/soft/parallel/f1_score_eq',
            'test/soft/intersectional/acc_eq']

INFO_COLS = ["data/name", "method/name", 'method/grid_size', "method/fairret", "method/statistic",
             "method/fairness_strength", "method/adv_dim", "method/reconstruction_strength",
             "data/main_sens_format"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=DATA_NAME, nargs='?',
                        help=f"Name of the data set. Current options are {DATASETS}")
    parser.add_argument('--notion', type=str, default=NOTION, nargs='?',
                        help=f"The fairness notion to be used. Current options are {list(FAIRNESS_NAMES.keys())}")
    parser.add_argument('--output_type', type=str, default=TYPE, nargs='?',
                        help="The output type. Either 'hard' or 'soft'")
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, nargs='?')
    args = parser.parse_args()

    data_results_path = os.path.join(args.results_dir, f"{args.data_name}.csv")
    try:
        df = pd.read_csv(data_results_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results for {args.data_name} not found. Should be on the repo.")

    data_names_loaded = df['data/name'].unique()
    assert data_names_loaded == np.array(args.data_name,)

    df_mean = df[NUM_COLS + INFO_COLS].groupby(INFO_COLS, dropna=False).mean().add_suffix('_mean')
    df_std = df[NUM_COLS + INFO_COLS].groupby(INFO_COLS, dropna=False).std().add_suffix('_std')
    df = pd.concat([df_mean, df_std], axis=1)
    df = df.reset_index()

    methods = pd.unique(df["method/name"])
    perf_measure = PERF_MEASURE_PER_OUTPUT_TYPE[args.output_type]

    naive_mean_violations = None
    if "no_method" in methods:
        naive_model_results = df[df["method/name"] == "no_method"]
        naive_mean = naive_model_results['test/' + perf_measure + '_mean'].item()
        naive_std = naive_model_results['test/' + perf_measure + '_std'].item()
        print(f"The naive model has an {naive_mean * 100}% {perf_measure} with std {naive_std * 100}")
        naive_mean_violations = {}
        for sens_attr in ["binary", "intersectional", "parallel"]:
            relevant_column = 'test/' + args.output_type + "/" + sens_attr + '/' + args.notion
            violation_mean = naive_model_results[relevant_column + "_mean"].item()
            naive_mean_violations[sens_attr] = violation_mean
            violation_std = naive_model_results[relevant_column + "_std"].item()
            print(f"The {args.notion} violation for {sens_attr} is {violation_mean} with std {violation_std}")

    methods = methods[methods != "no_method"]
    tabel = []
    for sens_attr in ["binary", "intersectional", "parallel"]:
        try:
            k_vals = K_TYPE_DICT[args.output_type][args.notion][sens_attr]
        except KeyError:
            if naive_mean_violations is None:
                raise ValueError("Either k values need to be defined, or a max k value needs to be deduced from naive")
            max_k_val = naive_mean_violations[sens_attr]
            max_k_val = np.round(max_k_val, 2)
            k_vals = [max_k_val / 4, max_k_val / 2, max_k_val]

        for k in k_vals:
            tabel_line = [SENS_ATTR_NAME[sens_attr], f'{k:.2f}']
            relevant_column = 'test/' + args.output_type + "/" + sens_attr + '/' + args.notion
            for method in methods:
                df_intermediate = df[df["method/name"] == method]
                df_intermediate = df_intermediate[df_intermediate[relevant_column + "_mean"] < k].reset_index(drop=True)
                if len(df_intermediate) > 0:
                    max_value_index = df_intermediate['test/' + perf_measure + '_mean'].argmax()
                    acc_value = df_intermediate['test/' + perf_measure + '_mean'][max_value_index]
                    std_value = df_intermediate['test/' + perf_measure + '_std'][max_value_index]
                    tabel_line.append(f'${acc_value * 100:.1f}_{{\pm {std_value * 100:.1f} }}$')
                else:
                    tabel_line.append("-")
            tabel.append(tabel_line)

    print(tabulate(tabel, headers=methods, tablefmt="latex_raw"))


if __name__ == '__main__':
    main()
