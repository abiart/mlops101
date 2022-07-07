from datetime import timedelta
import itertools
import json
from math import sqrt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml
import mlflow
from datetime import datetime
from time import time

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir.parent

# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))

# Custom functions and classes
# from sarima import *
from utils import *

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]

# Params contains all of the dataset creation parameters and model parameters
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)


# Step 1: Load the data, convert to a proper datetime format and apply correction
merged_data = pd.read_csv(
    os.path.join(
        proj_path, catalog["output_dir"]["dir"], catalog["output_dir"]["transactions"]
    )
)

merged_data["order_approved_at"] = pd.to_datetime(merged_data["order_approved_at"])
# merged_data['order_approved_at'] = merged_data['order_approved_at']

# Step 2: Create date folds
date_ranges = make_dates(params["olist"]["experiment_dates"])

for prod_cat in params["olist"]["product_categories"]:

    # Filter product category and dates
    df_filtered = merged_data[merged_data["product_category_name"] == prod_cat].copy()
    df_train = df_filtered[
        (df_filtered["order_approved_at"] >= date_ranges["train_start"].iloc[0])
        & (df_filtered["order_approved_at"] <= date_ranges["train_end"].iloc[-1])
    ]

    y_pred = df_train["payment_value"].values[:-1]
    y_true = df_train["payment_value"].values[1:]

    print(
        f"Training MAE product category {prod_cat}: {mean_absolute_error(y_true, y_pred)}"
    )
    print(sqrt(mean_squared_error(y_true, y_pred)))

    fdir = os.path.join(proj_path, catalog["results"]["dir"])
    fname = os.path.join(fdir, f"naive_training_{prod_cat}.csv")
    create_folder(fdir)

    pd.DataFrame(
        {
            "train_mae": [mean_absolute_error(y_true, y_pred)],
            "train_rmse": [sqrt(mean_squared_error(y_true, y_pred))],
        }
    ).to_csv(fname, index=False)
