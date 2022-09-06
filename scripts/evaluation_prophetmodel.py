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

    prophet_exp1 = prophet_exp1.rename(columns={"preds": "y_pred_prophet"})

    # Sets the index the same for all dfs so they can be concatenated based on the index

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
    return pd.DataFrame({"model": ["prophet"]})


def get_min_max(df, date_ranges):
    _metrics_prophet = []

    for window in date_ranges.itertuples():
        # Filter period
        temp = df[
            (pd.to_datetime(df.index) >= window[5])
            & (pd.to_datetime(df.index) <= window[6])
        ]

        _metrics_prophet.append(get_metrics(temp["y_true"], temp["y_pred_prophet"]))

    # Get the min and max for each metric for each model
    return pd.DataFrame()


metrics_df = pd.DataFrame()
metrics_mase_rmsse = pd.DataFrame()

for prod_cat in prod_categories:

    temp = all_product_categories[prod_cat]

    # Get metrics for each model

    metrics_prophet = get_metrics(temp["y_true"], temp["y_pred_prophet"])

    results = pd.DataFrame(
        [metrics_prophet],
        index=["prophet"],
    )
    results["product_category"] = prod_cat
    results = results.reset_index().rename(columns={"index": "model"})
    metrics_df = metrics_df.append(results).reset_index(drop=True)
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

met = metrics_df.set_index("product_category")
# metrics_df.to_csv("./data/04_results/metrics/metrics_prophetmodel.csv")
met.to_csv("./data/04_results/metrics/metrics_prophetmodel.csv")

# res = pd.DataFrame(metrics_df)

res = pd.DataFrame(met)
met.to_html("./data/04_results/metrics/summary.html")
