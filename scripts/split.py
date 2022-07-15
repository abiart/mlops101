import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import yaml

# Get the current project path (where you open the notebook)
# and go up two levels to get the project path
current_dir = Path.cwd()
proj_path = current_dir
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from dateutil.relativedelta import relativedelta

# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "src"))

# Custom functions and classes
from utils import make_dates, create_folder

# Catalog contains all the paths related to datasets
with open(os.path.join(proj_path, "conf/catalog.yml"), "r") as f:
    catalog = yaml.safe_load(f)["olist"]

# Params contains all of the dataset creation parameters and model parameters
with open(os.path.join(proj_path, "conf/params.yml"), "r") as f:
    params = yaml.safe_load(f)


date_ranges = make_dates(params["olist"]["experiment_dates"])


def show_windows(dates: pd.DataFrame, fname=None):
    """Generate a plot to view the time of the different folds.

    fname to save plot, specify a path.
    """
    # Register a date converter from pandas to work with matplotlib
    register_matplotlib_converters()

    mi = dates["train_start"].min() - relativedelta(weeks=1)
    ma = dates["test_end"].max() + relativedelta(weeks=1)

    plt.figure(figsize=(16, 5))
    plt.scatter(x=[mi, ma], y=[0, 0], alpha=0)
    for i, date_range in enumerate(dates.sort_values("train_start").itertuples()):

        # Width of bar in days
        train_len = (date_range[3] - date_range[1]).days
        valid_len = (date_range[4] - date_range[2]).days
        test_len = (date_range[6] - date_range[4]).days

        plt.barh(y=i, width=train_len, left=date_range[1], color="#C5CAE9")
        plt.barh(y=i, width=valid_len, left=date_range[3], color="#2196F3")
        plt.barh(y=i, width=test_len, left=date_range[5], color="#1A237E")
        plt.xticks(rotation=45)
        plt.ylabel("Fold Id")

    plt.title("Cross-Validation Through Time")
    plt.legend(["", "Train period", "Valid period", "Test period"])
    if fname:
        plt.savefig(fname)
    plt.show()


show_windows(date_ranges)
