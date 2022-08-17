from datetime import timedelta, datetime
import itertools
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
from pathlib import Path
from time import time


# check the version of python in vs , should be the V of the env
import mlflow
import mlflow.statsmodels
import yaml

# get url from dvc
# import dvc.api

# path (required)location and file name of the target to open, relative to the root of the project
# repo - specifies the location of the DVC project.
# rev - Git commit(any revision such as a branch or tag name, a commit hash or an experiment name


# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir


# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))


"""path = "data/02_processed/olist_sum_sales_with_payments.csv"
repo = "https://github.com/abiart/mlops101.git"
rev = "v2"

data_url = dvc.api.get_url(path, repo, rev)"""

from utils import *
from metrics import *
from sarima import *


# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]

# Params contains all of the dataset creation parameters and model parameters
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)

# Step 1: Read data
merged_data = pd.read_csv(
    os.path.join(
        proj_path, catalog["output_dir"]["dir"], catalog["output_dir"]["transactions"]
    )
)

merged_data["order_approved_at"] = pd.to_datetime(merged_data["order_approved_at"])
# merged_data['order_approved_at'] = merged_data['order_approved_at'] + timedelta(days=3)

# Step2: Create date folds
date_ranges = make_dates(params["olist"]["experiment_dates"])

mlflow.set_tracking_uri("http://ec2-44-208-155-40.compute-1.amazonaws.com:5000")
for prod_cat in params["olist"]["product_categories"]:
    print(f"Processing product category: {prod_cat}")

    # Initialize mlflow tracking

    mlflow.set_experiment(prod_cat)

    start_timer = time()
    lt_preds = []
    nd_preds = []
    used_params_folds = []
    for (
        _,
        train_start,
        train_end,
        valid_start,
        valid_end,
        test_start,
        test_end,
    ) in date_ranges.itertuples():

        # Filter product category and dates
        df_filtered = merged_data[
            merged_data["product_category_name"] == prod_cat
        ].copy()

        df_train = df_filtered[
            (df_filtered["order_approved_at"] >= train_start)
            & (df_filtered["order_approved_at"] <= train_end)
        ]
        df_valid = df_filtered[
            (df_filtered["order_approved_at"] >= valid_start)
            & (df_filtered["order_approved_at"] <= valid_end)
        ]
        df_test = df_filtered[
            (df_filtered["order_approved_at"] >= test_start)
            & (df_filtered["order_approved_at"] <= test_end)
        ]

        # Define set of parameters for SARIMA
        p = d = q = range(0, 1)
        pdq = list(itertools.product(p, d, q))
        spdq = list(itertools.product(p, d, q, [2, 3, 4]))
        all_params = list(itertools.product(pdq, spdq))

        model = SklearnSarima(df_train["payment_value"].values)
        model.fit_best_params(df_valid["payment_value"].values, all_params)

        lt_predictions = model.predict(df_test.shape[0])
        nd_predictions = model.fit_predict(df_test["payment_value"].values)

        lt_preds.extend(lt_predictions)
        nd_preds.extend(nd_predictions)

        used_params = model.get_params()
        aboutparams = [used_params]
        used_params_folds.append(used_params)

    df_filtered = merged_data[
        (merged_data["product_category_name"] == prod_cat)
        & (
            merged_data["order_approved_at"]
            >= params["olist"]["experiment_dates"]["test_start"]
        )
        & (
            merged_data["order_approved_at"]
            <= params["olist"]["experiment_dates"]["test_end"]
        )
    ].copy()

    lt_metrics = get_metrics(df_filtered["payment_value"].values, lt_preds)
    nd_metrics = get_metrics(df_filtered["payment_value"].values, nd_preds)
    # save result of model in data/results in name_of_categorie.csv
    fdir = os.path.join(proj_path, catalog["results"]["dir"])
    fname = os.path.join(fdir, f"exp1_sarima_{prod_cat}.csv")
    create_folder(fdir)

    save_data = pd.DataFrame(
        {
            "y_true": df_filtered["payment_value"].values,
            "nd_preds": np.array(nd_preds).flatten(),
            "lt_preds": lt_preds,
            "dates": df_filtered["order_approved_at"].values,
        }
    )
    save_data2 = save_data.set_index("dates")
    save_data2.to_csv(fname)
    duration_min = int((time() - start_timer) // 60)
    with mlflow.start_run() as run:
        mlflow.log_param("Product Category", prod_cat)
        mlflow.log_param("SARIMA_Params", aboutparams)
        # mlflow.log_param("data_url", data_url)
        # mlflow.log_param("data_version", version)
        mlflow.log_metrics(lt_metrics)
        mlflow.log_metrics(nd_metrics)
        mlflow.log_artifact(fname)

        mlflow.log_metric("time", duration_min)
        # mlflow.statsmodels.autolog()
        mlflow.sklearn.log_model(model, "sarmia_model")
