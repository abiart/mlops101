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
import mlflow
import yaml

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir.parent


# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))
from fbprophet import Prophet

from metrics import *
from utils import *

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

for prod_cat in params["olist"]["product_categories"]:
    print(f"Processing product category: {prod_cat}")

    # Initialize mlflow tracking
    create_folder(os.path.join(proj_path, "mlruns"))
    # mlflow.set_tracking_uri(os.path.join(proj_path, 'mlruns'))
    mlflow.set_tracking_uri(os.path.join("../../", "mlruns"))
    mlflow.set_experiment(prod_cat)
    metrics = []
    start_timer = time()
    all_predictions = []
    # Iterate over each period, unpack tuple in each variable.
    # in each of the period, we will find the best set of parameters,
    # which will represent the time-series cross validation methodology
    for (
        _,
        train_start,
        train_end,
        valid_start,
        valid_end,
        test_start,
        test_end,
    ) in date_ranges.itertuples():
        print(f"Processing range {str(train_start.date())} to {str(test_end.date())}")

        train_x = merged_data[
            (merged_data["order_approved_at"] >= train_start)
            & (merged_data["order_approved_at"] <= valid_end)
            & (merged_data["product_category_name"] == prod_cat)
        ][["order_approved_at", "payment_value"]]

        # Doesn't need a validation period.
        valid_x = merged_data[
            (merged_data["order_approved_at"] >= valid_start)
            & (merged_data["order_approved_at"] <= valid_end)
        ]
        test_y = merged_data[
            (merged_data["order_approved_at"] >= test_start)
            & (merged_data["order_approved_at"] <= test_end)
            & (merged_data["product_category_name"] == prod_cat)
        ][["order_approved_at", "payment_value"]]
        # Prophet expects two columns, one with the label 'ds' for the dates and y for the values
        train_x = train_x.rename(
            columns={"order_approved_at": "ds", "payment_value": "y"}
        )
        test_y = test_y.rename(
            columns={"order_approved_at": "ds", "payment_value": "y"}
        )

        # Iterate over the periods to make next-day forecasts
        predictions = []
        for i in range(test_y.shape[0]):

            # Instantiate a new Prophet object that represents the model
            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )

            # Call the built-in holiday collection for US to be included in the model
            model.add_country_holidays(country_name="BR")

            # Fit the FB Prohpet Model
            model.fit(pd.concat([train_x.iloc[i:], test_y.iloc[:i]]))
            future = model.make_future_dataframe(periods=1, freq="7D")
            fcst = model.predict(future)["yhat"].iloc[-1]
            predictions.append(fcst)

        all_predictions.extend(predictions)

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

    metrics = get_metrics(df_filtered["payment_value"].values, all_predictions)

    # store predictions
    fdir = os.path.join(proj_path, catalog["results"]["dir"])
    fname = os.path.join(fdir, f"exp1_prophet_{prod_cat}.csv")
    create_folder(fdir)

    save_data = pd.DataFrame(
        {
            "y_true": df_filtered["payment_value"].values,
            "preds": all_predictions,
            "dates": df_filtered["order_approved_at"],
        }
    )

    save_data.to_csv(fname)
    duration_min = int((time() - start_timer) // 60)

    with mlflow.start_run():
        mlflow.log_artifact(fname)
        mlflow.log_param("Product Category", prod_cat)
        mlflow.log_param("model", "prophet")
        mlflow.log_metrics(metrics)
        mlflow.log_metric("time", duration_min)
