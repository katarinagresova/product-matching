import numpy as np
import pandas as pd
from matching.model.tokenizer import words
from typing import List

def get_seller_warnings(left_seller: pd.DataFrame, left_seller_name: str, right_seller: pd.DataFrame, right_seller_name: str, cat_columns: List[str], str_columns: List[str], real_pairs: pd.DataFrame) -> pd.DataFrame:

    warnings = real_pairs[[left_seller_name, right_seller_name]].copy()
    
    for column_name in cat_columns:
        param_warnings = get_cat_warnings(left_seller, left_seller_name, right_seller, right_seller_name, column_name, real_pairs)
        warnings = warnings.merge(param_warnings, on=[left_seller_name, right_seller_name], how='left')
    
    for column_name in str_columns:
        param_warnings = get_str_warnings(left_seller, left_seller_name, right_seller, right_seller_name, column_name, real_pairs)
        warnings = warnings.merge(param_warnings, on=[left_seller_name, right_seller_name], how='left')

    # drop rows with NaN in all columns except url
    warnings = warnings[warnings.isnull().sum(axis=1) < 2*(len(cat_columns) + len(str_columns))]

    # count not null columns (divide by 2 because columns are duplicated and subtract one for url)
    warnings['count'] = warnings.count(axis=1) / 2 - 1

    return warnings.sort_values('count', ascending=False)


def get_cat_warnings(left_seller, left_seller_name, right_seller, right_seller_name, column_name, real_pairs): 

    left_seller_sub = left_seller[['url', column_name]]
    right_seller_sub = right_seller[['url', column_name]]

    left_seller_sub.columns = [left_seller_name, column_name + '_' + left_seller_name]
    right_seller_sub.columns = [right_seller_name, column_name + '_' + right_seller_name]

    merged = real_pairs.merge(left_seller_sub, on=left_seller_name, how='left').merge(right_seller_sub, on=right_seller_name, how='left')

    cond2 = (merged[column_name + '_' + left_seller_name] == merged[column_name + '_' + right_seller_name])
    cond1 = (merged[column_name + '_' + left_seller_name].isnull()) | (merged[column_name + '_' + right_seller_name].isnull())
    merged['match'] = np.select([cond1, cond2], [None, True], False)

    return merged[merged['match'] == False][[left_seller_name, right_seller_name, column_name + '_' + left_seller_name, column_name + '_' + right_seller_name]]


def get_str_warnings(left_seller, left_seller_name, right_seller, right_seller_name, column_name, real_pairs): 

    left_seller_sub = left_seller[['url', column_name]].copy()
    right_seller_sub = right_seller[['url', column_name]].copy()

    left_seller_sub[column_name] = left_seller_sub[column_name].apply(lambda x: words(x))
    right_seller_sub[column_name] = right_seller_sub[column_name].apply(lambda x: words(x))

    left_seller_sub.columns = [left_seller_name, column_name + '_' + left_seller_name]
    right_seller_sub.columns = [right_seller_name, column_name + '_' + right_seller_name]

    merged = real_pairs.merge(left_seller_sub, on=left_seller_name, how='left').merge(right_seller_sub, on=right_seller_name, how='left')

    cond2 = (merged[column_name + '_' + left_seller_name] == merged[column_name + '_' + right_seller_name])
    cond1 = (merged[column_name + '_' + left_seller_name].isnull()) | (merged[column_name + '_' + right_seller_name].isnull())
    merged['match'] = np.select([cond1, cond2], [None, True], False)

    merged['match'] = merged.apply(lambda x: any(set(x[column_name + '_' + left_seller_name]) & (set(x[column_name + '_' + right_seller_name])) if type(x[column_name + '_' + left_seller_name]) == list and type(x[column_name + '_' + right_seller_name]) == list else x), axis=1)

    return merged[merged['match'] == False][[left_seller_name, right_seller_name, column_name + '_' + left_seller_name, column_name + '_' + right_seller_name]]