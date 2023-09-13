import os
from loguru import logger

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import optuna

def plot_optimization_results(study, output_path='.'):
    """Plot the parameter importance, the contour, the EDF, the history and the slice for the optimization.

    Args:
        study (optuna study): study of the hyperparameters
        output_path (str, optional): path to save the plots to. Defaults to '.'.

    Returns:
        list: path of the written files from the working directory.
    """

    written_files=[]

    fig_importance = optuna.visualization.plot_param_importances(study)
    feature_path = os.path.join(output_path, 'importance.jpeg')
    # fig_importance.write_html(feature_path.replace('jpeg', 'html'))
    fig_importance.write_image(feature_path)
    written_files.append(feature_path)

    # fig_contour = optuna.visualization.plot_contour(study)
    # feature_path = os.path.join(output_path, 'contour.jpeg')
    # fig_contour.write_html(feature_path.replace('jpeg', 'html'))
    # # fig_contour.write_image(feature_path)
    # written_files.append(feature_path)

    fig_edf = optuna.visualization.plot_edf(study)
    feature_path = os.path.join(output_path, 'edf.jpeg')
    # fig_edf.write_html(feature_path.replace('jpeg', 'html'))
    fig_edf.write_image(feature_path)
    written_files.append(feature_path)

    fig_history = optuna.visualization.plot_optimization_history(study)
    feature_path = os.path.join(output_path, 'history.jpeg')
    fig_history.write_html(feature_path.replace('jpeg', 'html'))
    # fig_history.write_image(feature_path)
    written_files.append(feature_path)

    fig_slice = optuna.visualization.plot_slice(study)
    feature_path = os.path.join(output_path, 'slice.jpeg')
    fig_slice.write_html(feature_path.replace('jpeg', 'html'))
    # fig_slice.write_image(feature_path)
    written_files.append(feature_path)

    return written_files

def save_best_hyperparameters(study, output_dir='.'):
    """Save the best hyperparameters into a txt file.

    Args:
        study (optuna study): study of the hyperparameters
        output_dir (str, optional): path to save the txt file to. Defaults to '.'.

    Returns:
        str: feature path
    """

    best_trial = study.best_trial.number
    best_params = study.best_params
    best_val = study.best_value

    best_hyperparam_dict={'best_trial': best_trial, 'best_value': best_val}

    for key in best_params.keys():
        best_hyperparam_dict[key]=best_params[key]

    best_hyperparam_df=pd.DataFrame(best_hyperparam_dict, index=[0])
    feature_path=os.path.join(output_dir, 'best_hyperparams.csv')
    best_hyperparam_df.to_csv(feature_path, index=False, header=True)
    
    return feature_path