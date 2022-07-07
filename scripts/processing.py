import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime, timedelta
import sys


from sklearn import preprocessing

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir

# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)["olist"]


transactions_count = pd.read_csv(
    os.path.join(
        proj_path, catalog["output_dir"]["dir"], catalog["output_dir"]["transactions"]
    )
)
transactions_sum = pd.read_csv(
    os.path.join(
        proj_path,
        catalog["output_dir"]["dir"],
        catalog["output_dir"]["sum_transactions"],
    )
)

sort_by_count = transactions_count.groupby("product_category_name").sum()

sort_by_count.sort_values(by="payment_value", ascending=False)


def plot_data(data, ylabel="count"):

    sorted_prod = sort_by_count.sort_values(by="payment_value", ascending=False).index

    fig, axes = plt.subplots(nrows=24, ncols=3, figsize=(20, 5 * 24))
    plt.subplots_adjust(hspace=0.8)

    # For every product category, filter the data and plot it
    prod_i = 0
    for row in range(24):
        for col in range(3):
            filtered_data = data[
                (data["product_category_name"] == sorted_prod[prod_i])
                & (data["order_approved_at"] < params["experiment_dates"]["test_start"])
            ]

            filtered_data.plot(
                x="order_approved_at",
                y="payment_value",
                ax=axes[row, col],
                title=f"Product Category: {sorted_prod[prod_i]}",
                label=ylabel,
            )
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].set_xlabel("Date")

            for label in axes[row, col].get_xticklabels():
                label.set_rotation(40)
                label.set_horizontalalignment("right")
            prod_i += 1
            if prod_i >= len(sorted_prod):
                break
        if prod_i >= len(sorted_prod):
            break

    plt.show()


# plot_data(transactions_count, ylabel='Transaction Count')


def count_nulls_by_line(df):

    return df.isnull().sum().sort_values(ascending=False)


def null_percent_by_line(df):
    return (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)


def preprocess_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
