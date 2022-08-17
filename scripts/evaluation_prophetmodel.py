import matplotlib as plt

plt.rcParams.update({"figure.max_open_warning": 0})
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import yaml
from tqdm import tqdm

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir

# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))


import xgboost as xgb
from xgboost import XGBClassifier
from metrics import mean_absolute_percentage_error, get_metrics
from utils import make_dates
from utils import create_folder

# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]

# Params contains all of the dataset creation parameters and model parameters
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)

import mlflow

from glob import glob

from sklearn.metrics import r2_score


prod_categories = params["olist"]["product_categories"]
date_ranges = make_dates(params["olist"]["experiment_dates"])

# Aggregate all predictions of each model for exp1

all_product_categories = {}

base_mae_rmse_ll = []

for prod_cat in prod_categories:

    prophet_exp1 = pd.read_csv(
        proj_path / catalog["results"]["dir"] / f"exp1_prophet_{prod_cat}.csv"
    )

    # clean up each experiments and standardize their outputs here
    # xgb_exp1 = xgb_exp1.rename(columns={"preds": "y_pred_xgb"})
    # lstm_exp1 = lstm_exp1.rename(columns={'y_pred': 'y_pred_lstm'})
    # sarima_exp1 = sarima_exp1.rename(columns={'lt_preds': 'y_pred_lt_sarima'}) # can be lt_preds

    prophet_exp1 = prophet_exp1.rename(columns={"preds": "y_pred_prophet"})

    # Sets the index the same for all dfs so they can be concatenated based on the index
    # xgb_exp1.set_index("dates", inplace=True)
    # lstm_exp1.set_index('dates', inplace=True)
    # sarima_exp1.set_index("dates", inplace=True)
    prophet_exp1.set_index("dates", inplace=True)

    df = pd.concat(
        [
            prophet_exp1[["y_true", "y_pred_prophet"]],
        ],
        axis=1,
    )

    base_metrics = pd.read_csv(
        proj_path / catalog["results"]["dir"] / f"naive_training_{prod_cat}.csv"
    )

    all_product_categories[prod_cat] = df

    base_mae_rmse_ll.append(
        {
            "product_category": prod_cat,
            "base_mae": base_metrics["train_mae"].values[0],
            "base_rmse": base_metrics["train_rmse"].values[0],
        }
    )

base_mae_rmse = pd.DataFrame(base_mae_rmse_ll)


def get_min_max(df, date_ranges):

    _metrics_prophet = []

    for window in date_ranges.itertuples():
        # Filter period
        temp = df[
            (pd.to_datetime(df.index) >= window[5])
            & (pd.to_datetime(df.index) <= window[6])
        ]

        # _metrics_lstm.append(get_metrics(temp['y_true'], temp['y_pred_lstm']))
        _metrics_prophet.append(get_metrics(temp["y_true"], temp["y_pred_prophet"]))

    # Get the min and max for each metric for each model
    return pd.DataFrame(
        {
            "model": ["prophet"],
            "min_wape": [
                pd.DataFrame(_metrics_prophet)["wape"].min(),
            ],
            "min_rmse": [
                pd.DataFrame(_metrics_prophet)["rmse"].min(),
            ],
            "min_r2": [
                pd.DataFrame(_metrics_prophet)["r2"].min(),
            ],
            "min_mape": [
                pd.DataFrame(_metrics_prophet)["mape"].min(),
            ],
            "max_wape": [
                pd.DataFrame(_metrics_prophet)["wape"].max(),
            ],
            "max_rmse": [
                pd.DataFrame(_metrics_prophet)["rmse"].max(),
            ],
            "max_r2": [
                pd.DataFrame(_metrics_prophet)["r2"].max(),
            ],
            "max_mape": [
                pd.DataFrame(_metrics_prophet)["mape"].max(),
            ],
            "max_mae": [
                pd.DataFrame(_metrics_prophet)["mae"].max(),
            ],
            "min_mae": [
                pd.DataFrame(_metrics_prophet)["mae"].min(),
            ],
        }
    )


def get_min_max(df, date_ranges):

    # _metrics_xgb = []

    # _metrics_lstm = []
    _metrics_prophet = []

    for window in date_ranges.itertuples():
        # Filter period
        temp = df[
            (pd.to_datetime(df.index) >= window[5])
            & (pd.to_datetime(df.index) <= window[6])
        ]
        # _metrics_xgb.append(get_metrics(temp["y_true"], temp["y_pred_xgb"]))

        # _metrics_lstm.append(get_metrics(temp['y_true'], temp['y_pred_lstm']))
        _metrics_prophet.append(get_metrics(temp["y_true"], temp["y_pred_prophet"]))

    # Get the min and max for each metric for each model
    return pd.DataFrame(
        {
            "model": ["prophet"],
            "min_wape": [
                pd.DataFrame(_metrics_prophet)["wape"].min(),
                # pd.DataFrame(_metrics_xgb)["wape"].min(),
            ],
            "min_rmse": [
                pd.DataFrame(_metrics_prophet)["rmse"].min(),
            ],
            "min_r2": [
                pd.DataFrame(_metrics_prophet)["r2"].min(),
            ],
            "min_mape": [
                pd.DataFrame(_metrics_prophet)["mape"].min(),
            ],
            "max_wape": [
                pd.DataFrame(_metrics_prophet)["wape"].max(),
            ],
            "max_rmse": [
                pd.DataFrame(_metrics_prophet)["rmse"].max(),
            ],
            "max_r2": [
                pd.DataFrame(_metrics_prophet)["r2"].max(),
            ],
            "max_mape": [
                pd.DataFrame(_metrics_prophet)["mape"].max(),
            ],
            "max_mae": [
                pd.DataFrame(_metrics_prophet)["mae"].max(),
            ],
            "min_mae": [
                pd.DataFrame(_metrics_prophet)["mae"].min(),
            ],
        }
    )


metrics_df = pd.DataFrame()
metrics_mase_rmsse = pd.DataFrame()

for prod_cat in prod_categories:

    temp = all_product_categories[prod_cat]

    # Get metrics for each model
    # metrics_xgb = get_metrics(temp["y_true"], temp["y_pred_xgb"])

    # metrics_lstm = get_metrics(temp['y_true'], temp['y_pred_lstm'])
    metrics_prophet = get_metrics(temp["y_true"], temp["y_pred_prophet"])

    results = pd.DataFrame(
        [metrics_prophet],
        index=["prophet"],
    )
    results["product_category"] = prod_cat
    results = results.reset_index().rename(columns={"index": "model"})

    # Calculate the ranks for each metric
    results["rank_mape"] = results.rank(axis=0)["mape"]
    results["rank_wape"] = results.rank(axis=0)["wape"]
    results["rank_rmse"] = results.rank(axis=0)["rmse"]
    results["rank_mae"] = results.rank(axis=0)["mae"]
    results["rank_r2"] = results.rank(axis=0, ascending=False)["r2"]

    # Add rank for rmsse and mase
    results = results.merge(base_mae_rmse, how="left", on="product_category")
    results["mase"] = results["mae"] / results["base_mae"]
    results["rmsse"] = results["rmse"] / results["base_rmse"]
    results["rank_rmsse"] = results.rank(axis=0)["rmsse"]
    results["rank_mase"] = results.rank(axis=0)["mase"]

    min_max_df = get_min_max(df, make_dates(params["olist"]["experiment_dates"]))
    results = results.merge(min_max_df, how="inner", on="model")

    # Calculate the minimum and maximum of each fold.

    metrics_df = metrics_df.append(results).reset_index(drop=True)

    # merge base metrics for MASE and RMSSE

#     metrics_mase_rmsse = metrics_df.merge(base_mae_rmse, how='left', on='product_category')
#     metrics_mase_rmsse = metrics_mase_rmsse[['model', 'rmse', 'mae', 'product_category', 'base_mae', 'base_rmse']]
#     metrics_mase_rmsse['mase'] = metrics_mase_rmsse['mae'] / metrics_mase_rmsse['base_mae']
#     metrics_mase_rmsse['rmsse'] = metrics_mase_rmsse['rmse'] / metrics_mase_rmsse['base_rmse']

metrics_df[["model", "product_category", "mase", "rmsse", "rank_mase", "rank_rmsse"]]

all_dfs_mase_rmsse = []
for metric in ["rmsse", "mase"]:
    rank_df = pd.DataFrame(
        metrics_df.groupby("model")[f"rank_{metric}"]
        .value_counts()
        .rename(f"cnt_rank_{metric}")
    ).reset_index()
    rank_df[f"rank_{metric}"] = rank_df[f"rank_{metric}"].astype(int)
    all_dfs_mase_rmsse.append(
        rank_df.pivot_table(
            index="model", columns=f"rank_{metric}", values=f"cnt_rank_{metric}"
        ).add_prefix(f"{metric}_")
    )
pd.concat(all_dfs_mase_rmsse, axis=1)

all_dfs = []
for metric in ["mape", "rmse", "wape", "r2", "mae"]:
    rank_df = pd.DataFrame(
        metrics_df.groupby("model")[f"rank_{metric}"]
        .value_counts()
        .rename(f"cnt_rank_{metric}")
    ).reset_index()
    rank_df[f"rank_{metric}"] = rank_df[f"rank_{metric}"].astype(int)
    all_dfs.append(
        rank_df.pivot_table(
            index="model", columns=f"rank_{metric}", values=f"cnt_rank_{metric}"
        ).add_prefix(f"{metric}_")
    )

for prod_cat in prod_categories:

    temp = all_product_categories[prod_cat]
    temp.sort_index(inplace=True)

    for model in ["prophet"]:

        fig, axs = plt.subplots(2, 1, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.8)

        # Forecasts
        # axs[0].plot(temp["y_true"], marker="o", label="Real", alpha=0.8)
        if model == "sarima":
            # axs[0].plot(temp[f'y_pred_lt_{model}'], marker='o', label='Prediction (Long Term)',alpha=0.8)
            axs[0].plot(
                temp[f"y_pred_{model}"],
                marker="o",
                label="Prediction (Next Day)",
                alpha=0.8,
            )
        else:
            axs[0].plot(
                temp[f"y_pred_{model}"], marker="o", label="Prediction", alpha=0.8
            )
        axs[0].set_title(f"Model: {model} \nProduct Category: {prod_cat}")
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("Total Sales (Count)")
        axs[0].legend()
        for tick in axs[0].get_xticklabels():
            tick.set_rotation(40)
            tick.set_horizontalalignment("right")

        # Residuals
        axs[1].scatter(
            x=temp.index,
            y=(temp["y_true"] - temp[f"y_pred_{model}"]),
            marker="o",
            label="Residuals",
            alpha=0.8,
        )
        axs[1].plot([0] * temp.shape[0])
        axs[1].set_title(f"Model: {model} \nProduct Category: {prod_cat}")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Residuals")
        axs[1].legend()
        for tick in axs[1].get_xticklabels():
            tick.set_rotation(40)
            tick.set_horizontalalignment("right")

        # plt.show()

        # use the figure instance
        fig.savefig(
            "./data/04_results/plots/"
            f"Model: {model} Product Category: {prod_cat}" + ".png"
        )

    print("\n\n\n")

# print(metrics_df)
# metrics_df[['model', 'product_category', 'mase', 'rmsse', 'rank_mase','rank_rmsse']]
# converting to csv

metrics_df.to_csv("./data/04_results/metrics/metrics_prophetmodel.csv")


res = pd.DataFrame(metrics_df)
# dfj=metrics_df[['mase', 'rmsse']]
res.to_json("./data/04_results/metrics/summary.json")
