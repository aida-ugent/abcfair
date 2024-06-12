import pandas as pd
import os
import wandb

from utils import RESULTS_DIR, ENTITY, PROJECT


SWEEP_IDS = [
    "s2v0enhs",  # All of prevalence sampling
    "ms6woq8a",  # Extra prevalence sampling
    "m1pch8u0",  # All of data repairer
    "oxezhbrx",  # Extra data repairer
    "98r2w8h4",  # All of label flipping
    "b29g81rp",  # Extra label flipping
    "o2khawsq",  # All of LFR
    "s5uvr62j",  # Extra LFR
    "209dth9l",  # All of Fairret
    "508p41l0",  # Extra fairret KL-Proj
    "rl0yq05m",  # Extra fairret norm
    "58g269ge",  # All of LAFTR
    "e4b7ruvf",  # Extra LAFTR
    "jw0sa3s4",  # All of Prejudice remover
    "vcfg1cvp",  # Extra prejudice remover
    "lyhn65t2",  # All of Error Parity
    "ipixdyn8",  # Extra Error parity
    "nm9yg4s0",  # All of Reductions
    "tbfcj0ta",  # Extra LFR runs
    "bufvheyo",  # Reductions SchoolBiased
    "e7iiiqko",  # Extra reductions
    "m46d4xqu",  # Fairret SchoolBiased
    "dubmupqx",  # Naive
    "4wpsvwkp",  # Extra naive
]

SWEEP_IDS_2 = [
    "m1pch8u0",  # All of data repairer
    "rcraog71",  # Extra runs data_repairer
    "t4g7nmbw",  # Extra extra data repairer
    "98r2w8h4",  # All of label flipping
    "gpu1yxsi",  # Extra label flipping
    "o2khawsq",  # All of LFR
    "tbfcj0ta",  # Extra LFR runs
    "iiao3mu7",  # Extra extra LFR runs
    "s2v0enhs",  # All of prevalence sampling
    "121zelw3",  # Extra prevalence sampling
    "6dgjwpyw",  # Fairret norm
    "8vqbnn07",  # Extra fairret norm
    "wnnp0tvv",  # Fairret KL
    "b3axarys",  # Actually fairret KL
    "g5iot8ry",  # Extra fairret KL
    "qovwf3o0",  # LAFTR
    "yzmwd5vb",  # Extra LAFTR
    "xsj4kc9y",  # Prejudice remover
    "j4likjnd",  # Extra prejudice remover
    "1smzywc7",  # Reductions
    "x9zyml20",  # Extra reductions
    "piz7nfjx",  # Error parity
    "mgjsp7ne",  # Extra error parity
]


def download_sweep_results(sweep_id, api):
    series = []
    for run in api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}").runs:
        config = run.config
        config['sweep_id'] = run.sweep.id
        summary = dict(run.summary)
        series_all_info = pd.concat([pd.json_normalize(config, sep='/').squeeze(), pd.Series(summary, name=run.name)])
        if series_all_info["method/name"] == "fairret":
            series_all_info["method/name"] = series_all_info["method/name"] + "_" + series_all_info["method/fairret"]
        series.append(series_all_info)

    df = pd.DataFrame(series)
    df = df.drop(['num_threads', '_step', '_timestamp', 'data_config'], axis=1)
    return df


def main():
    # api = wandb.Api()
    # all_dfs = []
    #
    # sweep_ids = list(set(SWEEP_IDS + SWEEP_IDS_2))
    #
    # for sweep_id in sweep_ids:
    #     df = download_sweep_results(sweep_id, api)
    #     all_dfs.append(df.reset_index(drop=True))
    # df = pd.concat(all_dfs, ignore_index=True)

    # for data_name, group_df in df.groupby('data/name'):
    #     group_df.to_csv(os.path.join(RESULTS_DIR, f"{data_name}.csv"), index=False)

    new_sens_attr_names = {
        'one_sens': 'binary',
        'intersect_sens': 'intersectional',
        'all_sens': 'parallel'
    }

    files_in_results = os.listdir(RESULTS_DIR)
    for file in files_in_results:
        df = None
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        else:
            ValueError(f"File {file} is not a csv file.")

        # cols_to_drop = ['sweep_id', 'model_options/complex_model', "_wandb"]
        # cols_to_rename = {
        #     'model_options/sensitive_attributes': 'data/main_sens_format',
        # }
        # for col in df.columns:
        #     split_col = col.split('/')
        #     if split_col[0] in ['training', 'val', 'test']:
        #         if split_col[1] in ['hard', 'soft']:
        #             new_sens_attr_name = new_sens_attr_names[split_col[2]]
        #             cols_to_rename[col] = f"{split_col[0]}/{split_col[1]}/{new_sens_attr_name}/{split_col[3]}"
        # df = df.drop(cols_to_drop, axis=1)
        # df = df.rename(columns=cols_to_rename)
        # print(df.columns)

        df = df.replace({'one_sens_attr': 'binary', 'intersectional_sens_groups': 'intersectional', 'diff_sens_groups': 'parallel'})
        df = df.replace({
            'ACSPublicCoverageProcessed': 'ACSPublicCoverage',
            'ACSEmploymentProcessed': 'ACSEmployment',
            'ACSIncomeProcessed': 'ACSIncome',
            'ACSMobilityProcessed': 'ACSMobility',
            'ACSTravelTimeProcessed': 'ACSTravelTime',
        })

        df.to_csv(os.path.join(RESULTS_DIR, file), index=False)


if __name__ == '__main__':
    main()
