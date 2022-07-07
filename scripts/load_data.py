import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir

# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))

# from utils import create_folder
# from date_utils import filter_date_state

# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]

# Params contains all of the dataset creation parameters and model parameters
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)


# Read the raw data from the tables
customers = pd.read_csv(
    proj_path / catalog["base_dir"] / catalog["tables"]["customers"]
)
products = pd.read_csv(proj_path / catalog["base_dir"] / catalog["tables"]["products"])
pc_name_trans = pd.read_csv(
    proj_path / catalog["base_dir"] / catalog["tables"]["pc_name_trans"]
)
orders = pd.read_csv(proj_path / catalog["base_dir"] / catalog["tables"]["orders"])
order_items = pd.read_csv(
    proj_path / catalog["base_dir"] / catalog["tables"]["orderitems"]
)
order_payments = pd.read_csv(
    proj_path / catalog["base_dir"] / catalog["tables"]["orderpayments"]
)

# Store product categories' translations in a dictionary
# and translate the product category column in the products table to english
pc_name_trans = pc_name_trans.set_index("product_category_name")[
    "product_category_name_english"
].to_dict()
products["product_category_name"] = products["product_category_name"].map(pc_name_trans)
# Join tables together
sales_order = pd.merge(orders, customers, on="customer_id", how="inner")
sales_order_item = order_items.merge(sales_order, on="order_id", how="left")
sales_order_full = sales_order_item.merge(products, on="product_id", how="inner")
sales_with_payments = sales_order_full.merge(order_payments, on="order_id")

# Convert date column to datatime object
sales_with_payments["order_approved_at"] = pd.to_datetime(
    sales_with_payments["order_approved_at"], format="%Y-%m-%d"
)

sales_with_payments.set_index("order_approved_at", inplace=True)


def create_folder(folder):
    "Utility function to create folder if it doesn't exist."
    if not os.path.exists(folder):
        os.makedirs(folder)


# Group sales togather, either by week or day
group_freq = "W"  # 'd' for day

sales_with_payments_count = (
    sales_with_payments.groupby(["product_category_name"])
    .resample(group_freq)["payment_value"]
    .count()  # sum
    .reset_index()
)
create_folder(proj_path / catalog["output_dir"]["dir"])
sales_with_payments_count.to_csv(
    proj_path / catalog["output_dir"]["dir"] / catalog["output_dir"]["transactions"]
)


group_freq = "W"  # 'd' for day

sales_with_payments_sum = (
    sales_with_payments.groupby(["product_category_name"])
    .resample(group_freq)["payment_value"]
    .sum()  # sum
    .reset_index()
)

create_folder(proj_path / catalog["output_dir"]["dir"])
sales_with_payments_sum.to_csv(
    proj_path / catalog["output_dir"]["dir"] / catalog["output_dir"]["sum_transactions"]
)


# How many of those categories do we have zero sales units
(
    sales_with_payments_count[
        (
            sales_with_payments_count["product_category_name"].isin(
                params["olist"]["product_categories"]
            )
        )
        & (
            sales_with_payments_count["order_approved_at"]
            >= params["olist"]["experiment_dates"]["train_start"]
        )
        & (
            sales_with_payments_count["order_approved_at"]
            <= params["olist"]["experiment_dates"]["test_end"]
        )
    ]
    .groupby("product_category_name")
    .agg(lambda x: x.eq(0).sum())
)
