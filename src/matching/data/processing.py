import pandas as pd
import numpy as np
import re


def get_unmatched(product_series_1, product_series_2):

    return pd.concat([
        product_series_1[~product_series_1.isin(product_series_2)].dropna(),
        product_series_2[~product_series_2.isin(product_series_1)].dropna()
    ]).values

def remove_unmatched(seller, seller_name, seller_id_column, pairs):

    unmatched = get_unmatched(seller[seller_id_column], pairs[seller_name])
    pairs.drop(pairs.loc[pairs[seller_name].isin(unmatched)].index, inplace=True)
    seller.drop(seller.loc[seller[seller_id_column].isin(unmatched)].index, inplace=True)

    return seller, pairs

def to_numeric(x):
    x = str(x).replace(',', '.')
    x = re.sub("[^0-9\.]", "", x)
    if x == '':
        return np.nan
    else:
        return round(float(x), 1)

def get_bool_columns(df):
    return [col for col in df.columns if df[col].isin(['True', 'true', 'False', 'false', 'nan']).all()]

def map_boolean(x, column):
    if x == 'True' or x == 'true':
        return column
    elif x == 'False' or x == 'false':
        return 'not_' + column
    else:
        return x

def make_mapping(list_of_sources, cai_column, ean_column):

    def merge_dicts_remove_duplicates(d1, d2):

        result = d1
        for k, v in d2.items():
            if k not in result.keys():
                result[k] = v
            else:
                if result[k] != v:
                    result[k] = np.nan
        return result

    mapping = {}
    for source in list_of_sources:

        temp_mapping = dict(zip(source[cai_column], source[ean_column]))
        mapping = merge_dicts_remove_duplicates(mapping, temp_mapping)
        
    mapping = {k: mapping[k] for k in mapping if not pd.isna(mapping[k])}

    return mapping

