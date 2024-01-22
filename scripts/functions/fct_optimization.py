import os
from loguru import logger

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import optuna


def plot_optimization_results(study, targets={0: 'f1 score'}, output_path='.'):
    """Plot the parameter importance, the contour, the EDF, the history and the slice for the optimization.

    Args:
        study (optuna study): study of the hyperparameters.
        targets (dict, optional): targets to optimize. Defaults to {0: 'f1 score'}.
        output_path (str, optional): path to save the plots to. Defaults to '.'.

    Returns:
        list: path of the written files from the working directory.
    """

    written_files = []

    for target in targets.keys():

        fig_edf = optuna.visualization.plot_edf(study, target=lambda t: t.values[target], target_name=targets[target])
        feature_path = os.path.join(output_path, f'edf_{targets[target]}.jpeg')
        fig_edf.write_image(feature_path)
        written_files.append(feature_path)

        fig_history = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[target], target_name=targets[target])
        feature_path = os.path.join(output_path, f'history_{targets[target]}.jpeg')
        fig_history.write_html(feature_path.replace('jpeg', 'html'))
        # fig_history.write_image(feature_path)
        written_files.append(feature_path)

        fig_slice = optuna.visualization.plot_slice(study, target=lambda t: t.values[target], target_name=targets[target])
        feature_path = os.path.join(output_path, f'slice_{targets[target]}.jpeg')
        fig_slice.write_html(feature_path.replace('jpeg', 'html'))
        # fig_slice.write_image(feature_path)
        written_files.append(feature_path)

        fig_importance = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[target], target_name=targets[target])
        feature_path = os.path.join(output_path, f'importance_{targets[target]}.jpeg')
        fig_importance.write_image(feature_path)
        written_files.append(feature_path)

    return written_files


def save_best_hyperparameters(study, targets={0: 'f1 score'}, output_dir='.'):
    """Save the best hyperparameters into a txt file.

    Args:
        study (optuna study): study of the hyperparameters.
        targets (dict, optional): targets to optimize. Defaults to {0: 'f1 score'}.
        output_dir (str, optional): path to save the txt file to. Defaults to '.'.

    Returns:
        str: feature path
    """

    try:
        best_trial = study.best_trial.number
        best_params = study.best_params
        best_val = study.best_value

        best_hyperparam_dict = {'best_trial': best_trial, 'best_value': best_val}

        for key in best_params.keys():
            best_hyperparam_dict[key] = best_params[key]

        best_hyperparam_df = pd.DataFrame(best_hyperparam_dict, index=[0])

    except RuntimeError as e:
        if 'A single best trial cannot be retrieved from a multi-objective study.' in str(e):

            trials_df = study.trials_dataframe()
            numbers_best_trials = [trial.number for trial in study.best_trials]
            best_hyperparam_df = trials_df[trials_df.number.isin(numbers_best_trials)].copy()

            for target in targets.keys():
                best_hyperparam_df.rename(columns={f'values_{target}':targets[target]}, inplace=True)
            
            best_hyperparam_df.drop(columns=['datetime_start', 'datetime_complete', 'duration', 'state'], inplace=True)

            for col in best_hyperparam_df.columns:
                str_to_remove = 'params'
                if col.startswith(str_to_remove):
                    best_hyperparam_df.rename(columns={col: col.lstrip(str_to_remove + '_')}, inplace=True)

    feature_path = os.path.join(output_dir, 'best_hyperparams.csv')
    best_hyperparam_df.to_csv(feature_path, index=False, header=True)
    
    return feature_path