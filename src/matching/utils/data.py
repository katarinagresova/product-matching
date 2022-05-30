import pandas as pd

EXCLUDED_COLUMNS = ['stripped_text', 'url_hash', 'availability', 'description', 'price']
def load_data(csv_path, excluded_columns=EXCLUDED_COLUMNS):

    headers = list(pd.read_csv(csv_path, nrows=1))

    return pd.read_csv(
        csv_path,
        usecols=[item for item in headers if item not in excluded_columns],
        dtype=str
    )