import sys

sys.path.append('.')  # noqa

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import (RESULTS_DIR, FAIRNESS_NAMES, PERF_NAMES, plot_accuracy_fairness_tradeoff,
                   PERF_MEASURE_PER_OUTPUT_TYPE, DATASETS)
from dataset.base import SENS_FORMATS

# DATA_NAME = "SchoolPerformanceBiased"
DATA_NAME = "SchoolPerformanceUnbiased"
NOTION = "forp"
TYPE = "hard"
SENS_ATTR = "binary"
METHODS = ['fairret_norm', 'fairret_KL_proj', 'laftr', 'prejudice_remover', 'prevalence_sampling',
           'error_parity']

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
    parser.add_argument('--sens_attr', type=str, default=SENS_ATTR, nargs='?',
                        help=f"The sensitive attribute format. Current options are {SENS_FORMATS}")
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, nargs='?')
    args = parser.parse_args()

    data_results_path = os.path.join(RESULTS_DIR, f"{args.data_name}.csv")
    df = pd.read_csv(data_results_path)
    data_names_loaded = df['data/name'].unique()
    assert data_names_loaded == np.array(args.data_name,)

    df = df[df["data/main_sens_format"] == args.sens_attr]
    df = df[df["method/name"].isin(METHODS)]
    perf_measure = PERF_MEASURE_PER_OUTPUT_TYPE[args.output_type]

    fig, ax = plt.subplots()
    relevant_fairness = 'test/' + args.output_type + "/" + args.sens_attr + '/' + args.notion
    relevant_performance = 'test/' + perf_measure

    STYLE = 'method/name'
    plot_accuracy_fairness_tradeoff(ax, df, relevant_fairness, relevant_performance, STYLE, INFO_COLS,
                                    order=None, x_range=None, n_std=1.0)
    plt.ylabel(PERF_NAMES[perf_measure])
    plt.xlabel(f"Violation of {args.output_type} {FAIRNESS_NAMES[args.notion]} for the {args.sens_attr} S format")
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
