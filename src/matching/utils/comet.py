import comet_ml
import pandas as pd
from matching.utils.data import load_data
import sys

def profile_datasets(name, csv_paths):

    experiment = comet_ml.Experiment(
        api_key="0VtQAtk0Chqw5xBqLdPD5d3y0",
        project_name="dataset-profile",
        workspace="proficio",
    )
    experiment.set_name(name)
    experiment.add_tag("profile")

    for csv_path in csv_paths:
        df = load_data(csv_path, excluded_columns=[]) 
        experiment.log_dataframe_profile(df, csv_path)

    experiment.end()

if __name__ == "__main__":
    profile_datasets(sys.argv[1], sys.argv[2:])