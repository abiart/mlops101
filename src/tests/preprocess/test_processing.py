import io
import builtins
import pytest
from pathlib import Path
import pandas as pd
import sys
import os

current_dir = Path.cwd()
proj_path = current_dir

# make the code in src available to import in this notebook
import sys

sys.path.append(os.path.join(proj_path, "scripts"))


# Preprocess Python file
import processing


# Parent Folder which contains test_data/
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


FILE_NAME = "test_olist_customers_dataset"
DATA_PATH = (
    os.path.dirname(os.path.realpath(__file__)) + "/test_data/" + FILE_NAME + ".csv"
)

# PROCESSED_DATA_PATH in my case :

PROCESSED_DATA_PATH = (
    os.path.dirname(os.path.realpath(__file__))
    + "/test_data/"
    + FILE_NAME
    + "_processed.csv"
)


def test_count_nulls_by_line():
    # Tests function that counts number of nulls by line on a dataframe
    data = pd.DataFrame([[0, 2], [0, 1], [6, None]])
    assert processing.count_nulls_by_line(data).to_list() == [1, 0]


def test_null_percent():
    # Tests function that gets the percentage of nulls by line on a dataframe
    data = pd.DataFrame([[0, 2], [1, None]])
    assert processing.null_percent_by_line(data).to_list() == [0.5, 0]


"""@pytest.mark.dependency()
def test_processing():
    # Checks if running the preprocess function returns an error
    processing.preprocess_data(DATA_PATH)"""


"""

@pytest.mark.dependency(depends=["test_processing"])
def test_processed_file_created():
    #  Checks if the processed file was created during test_preprocess() and is accessible
    f = open(PROCESSED_DATA_PATH)


@pytest.mark.dependency(depends=["test_processed_file_created"])
def test_processed_file_format():
    # Checks if the processed file is in  the correct format (.csv) and can be transformed in dataframe
    try:
        pd.read_csv(PROCESSED_DATA_PATH)
    except:
        raise RuntimeError("Unable to open " + PROCESSED_DATA_PATH + " as dataframe")
"""


"""@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    # Runs tests then cleans up the processed file
    yield
    try:
        os.remove(PROCESSED_DATA_PATH)
    except:
        pass"""
