import numpy as np
import pandas as pd


def make_matches_structure(left_seller_ids: pd.Series, left_seller_name: str, right_seller_name: str) -> pd.DataFrame:
    """Creates structure for holding best matches and their similarities.

    Represented as pd.DataFrame with columns [left_seller_name, left_seller_name, 'sim'].
    Values in left_seller_name column are unique identifiers of left seller.
    Values in left_seller_name column are np.nan. Identifiers of best matching right seller will be added here.
    Values in 'sim' column are zeros. Similarity scores of best matching right seller will be added here.

    Args:
        left_seller_ids (pd.Series): Unique identifiers of products of left seller.
        left_seller_name (str): Name of left seller.
        right_seller_name (str): Name of right seller.

    Returns:
        pd.DataFrame: Structure for holding best matches and their similarities
    """
    matches = pd.DataFrame(
        np.array([
            left_seller_ids, 
            np.empty(len(left_seller_ids)), 
            np.zeros(len(left_seller_ids))]).T, 
        columns=[left_seller_name, right_seller_name, 'sim'])
    matches[right_seller_name][:] = np.nan 
    
    return matches


def evaluate(matches, real_pairs, left_column_name, right_column_name):
    """ Evaluates, if predicted matches are corresponding to real pairs.

    Args:
        matches (DataFrame): predicted pairs in DataFrame with columns [left_side, right_side, similarity].
        real_pairs (DataFrame): DataFrame, where rows represent one product matched accross multiple sellers. 
        left_column_name (string): Name of seller 'left_side' in DataFrame 'real_pairs'.
        right_column_name (string): Name of seller 'right_side' in DataFrame 'real_pairs'.

    Returns:
        tuple(int, list, list): number of correct matches, list of correct matches, list of wrong matches
    """

    good = 0
    matched = []
    wrong_matched = []

    # our goal was to find correct pairs for all items in 'pneuboss_sample'
    for i in range(len(matches)):

        # pneuboss url
        left = matches.iloc[i][left_column_name]
        # we can find correct pair in original data
        right = real_pairs.loc[real_pairs[left_column_name] == left][right_column_name].values[0]

        # get the url it was matched to
        predicted_right = matches.loc[matches[left_column_name] == left][right_column_name].values[0]

        # now we have four options:
        # 1: pneuboss url was matched to right url
        # 2: pneuboss url was matched to wrong url (or it shouldn't be matched at all)
        # 3: pneuboss url wasn't matched and it is correct
        # 4: pneuboss url wasn't matched and should be

        # pneuboss url was matched
        if predicted_right is not np.nan:            

            # 1: pneuboss url was matched to right url
            if right == predicted_right:
                good += 1
                matched.append(matches.loc[matches[left_column_name] == left].values[0].copy())
            # 2: pneuboss url was matched to wrong url (or it shouldn't be matched at all)
            else:
                wrong_matched.append(matches.loc[matches[left_column_name] == left].values[0].copy())

        #pneuboss url wasn't matched
        else:
            # 3: pneuboss url wasn't matched and it is correct
            if right is np.nan:
                good += 1
                matched.append(np.array([left, np.nan, 0]))
            # 4: pneuboss url wasn't matched and should be
            else:
                wrong_matched.append(np.array([left, np.nan, 0]))

    return good, matched, wrong_matched


def add_labels(matches, real_pairs, left_column_name, right_column_name):
    """ Evaluates, if predicted matches are corresponding to real pairs.

    Args:
        matches (DataFrame): predicted pairs in DataFrame with columns [left_side, right_side, similarity].
        real_pairs (DataFrame): DataFrame, where rows represent one product matched accross multiple sellers. 
        left_column_name (string): Name of seller 'left_side' in DataFrame 'real_pairs'.
        right_column_name (string): Name of seller 'right_side' in DataFrame 'real_pairs'.

    Returns:
        tuple(int, list, list): number of correct matches, list of correct matches, list of wrong matches
    """

    label = []

    # our goal was to find correct pairs for all items in 'pneuboss_sample'
    for i in range(len(matches)):

        # pneuboss url
        left = matches.iloc[i][left_column_name]
        # we can find correct pair in original data
        right = real_pairs.loc[real_pairs[left_column_name] == left][right_column_name].values[0]
        if right is np.nan:
            label.append(0)
        else:
            label.append(1)

    matches['label'] = label

    return matches

def get_metrics(matches, real_pairs, left_column_name, right_column_name):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(matches)):

        # pneuboss url
        left = matches.iloc[i][left_column_name]
        # we can find correct pair in original data
        right = real_pairs.loc[real_pairs[left_column_name] == left][right_column_name].values[0]

        # get the url it was matched to
        predicted_right = matches.loc[matches[left_column_name] == left][right_column_name].values[0]

        # now we have four options:
        # 1: pneuboss url was matched to right url
        # 2: pneuboss url was matched to wrong url (or it shouldn't be matched at all)
        # 3: pneuboss url wasn't matched and it is correct
        # 4: pneuboss url wasn't matched and should be

        # pneuboss url was matched
        if predicted_right is not np.nan:            

            # 1: pneuboss url was matched to right url
            if right == predicted_right:
                tp += 1
            # 2: pneuboss url was matched to wrong url (or it shouldn't be matched at all)
            else:
                fp += 1

        #pneuboss url wasn't matched
        else:
            # 3: pneuboss url wasn't matched and it is correct
            if right is np.nan:
                tn += 1
            # 4: pneuboss url wasn't matched and should be
            else:
                fn += 1

    return tp, tn, fp, fn
