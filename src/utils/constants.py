import os
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

COLOR_PALETTE = sns.color_palette('tab10')
COLORS = [color for color in COLOR_PALETTE]
COLORMAP = {
    "no_method": COLORS[3],
    "LearningFairRepr": COLORS[2],
    "prevalence_sampling": COLORS[1],
    "fairret_KL_proj": COLORS[0],
    "data_repairer": COLORS[4],
    "label_flipping": COLORS[5],
    "error_parity": COLORS[6],
    "exp_gradient": COLORS[7],
    "laftr": COLORS[8],
    "prejudice_remover": COLORS[2],
    "red": COLORS[6],
    "fairret_norm": COLORS[7],
}

METHOD_NAMES = {'data_repairer': "Data Repairer", 'fairret_norm': "Fairret Norm",
                'fairret_KL_proj': "Fairret $KL_{Proj}$",
                'laftr': "LAFTR", 'prejudice_remover': "Prejudice Remover",
                'prevalence_sampling': "Prevalence Sampling",
                'error_parity': "Error Parity"}

FAIRNESS_NAMES = {
    "dem_par": "demographic parity",
    "eq_opp": "equal opportunity",
    "forp": "false omission rate parity",
    "pred_par": "predictive parity",
    "acc_eq": "accuracy equality",
    "f1_score_eq": "F1-score equality",
    "pred_eq": "predictive equality",
}

MARKERS = {
    "no_method": "X",
    "LearningFairRepr": "o",
    "prevalence_sampling": "P",
    "fairret_KL_proj": "^",
    "data_repairer": "s",
    "label_flipping": "D",
    "error_parity": "h",
    "exp_gradient": "v",
    "laftr": "d",
    "prejudice_remover": "v",
    "red": "h",
    "fairret_norm": "H",
}

PERF_NAMES = {"accuracy": "Accuracy", "auroc": "AUROC", "f1score": "F1-score"}

PERF_MEASURE_PER_OUTPUT_TYPE = {
    "hard": "accuracy",
    "soft": "auroc"
}

DATASETS = ['ACSPublicCoverage', 'ACSEmployment', 'ACSIncome', 'ACSMobility', 'ACSTravelTime',
            'SchoolPerformanceBiased', 'SchoolPerformanceUnbiased']
