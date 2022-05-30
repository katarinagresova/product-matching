import pandas as pd
import numpy as np
from matching.model.tokenizer import words
from matching.model.transform import tfidf
from matching.model.similarity import cossim_top
from matching.model.matcher import best_unique

class MatchingModel:
    """Wrapper for components that do functionality of a matching model.
        """

    def __init__(self, tokenizer=None, transform=None, similarity=None, matcher=None) -> None:
        """Initializing components of a model.

        Args:
            tokenizer (Callable): a function that takes a string as an input, and splits it into list of string tokens. 
                Default: model.tokenizer.words
            transform (Callable): a function that takes two pd.Series of strings and tokenizer as inputs, and both pd.Series converts into sparse matrices.
                Default: model.transform.tfidf
            similarity (Callable): a function that takes two sparse matrices (and some additional parameters), 
                and computes similarity between all rows from first matrix and all rows from second matrix. 
                Results are also returned as sparse matrix.
                Default: model.similarity.cossim_top
            matcher (Callable): a function that takes sparse matrix of similarities and two pd.Series of ids of input data,
                and returns best matching pairs and their matching score.
                Default: model.matcher.best_unique
        """

        self.tokenizer = words if tokenizer is None else tokenizer
        self.transform = tfidf if transform is None else transform
        self.similarity = cossim_top if similarity is None else similarity
        self.matcher = best_unique if matcher is None else matcher

        # Structure for holding matched products. Will be initialized in self.fit() method according to provided data.
        self.right_seller_name = None

    def fit(self, left_seller, left_seller_name, right_seller, right_seller_name) -> None:
        """Saves product data and initializes needed structures.

        Args:
            sellers (dict): Dictionary of product data for each seller
            main_seller (str): Name of the main seller - all other sellers will be matched to this one
        """

        self._validate_data()

        self.left_seller = left_seller.copy()
        self.right_seller = right_seller.copy()
        self.left_seller_name = left_seller_name
        self.right_seller_name = right_seller_name

        self._make_matches_structure()


    def predict(self, lower_bound=0.5, do_ean_check=True, exact_match_columns=[], exact_match_columns_with_nan=[]) -> dict:
        """Predicts the best pairs of products based on id exact match, dividing into groups and string matching within the groups.

        Args:
            lower_bound (float, optional): Only pairs with similarity higher then lower_bound are kept for further processing. Defaults to 0.5.

        Returns:
            dict: Dictionary of pd.DataFrame for each seller except for main seller. 
                pd.DataFrame has columns [main seller, other seller, similarity] and contains the best pairs of products and their matching score
        """
        
        self.predict_ean()
        self.predict_string(lower_bound, do_ean_check, exact_match_columns, exact_match_columns_with_nan)

        return self.matches


    def predict_ean(self) -> None:

        for left_seller_id in self.left_seller['id'].values:
            ean = self.left_seller[self.left_seller['id'] == left_seller_id]['ean'].values[0]
            if ean != np.nan and ean in self.right_seller['ean'].values:
                right_seller_id = self.right_seller[self.right_seller['ean'] == ean]['id'].values[0]
                self.matches.loc[self.matches[self.left_seller_name] == left_seller_id, [self.right_seller_name, 'sim']] = right_seller_id, 1

        self._drop_matched()

    def predict_string(self, lower_bound, do_ean_check, exact_match_columns, exact_match_columns_with_nan) -> None:
        
        def mark_matches(match) -> None:
            for index in match.index.values:
                left_seller_id = match.loc[index]['left_side']
                right_seller_id = match.loc[index]['right_side']
                sim = match.loc[index]['similarity']

                self.matches.loc[self.matches[self.left_seller_name] == left_seller_id, [self.right_seller_name, 'sim']] = right_seller_id, sim

        def compute_matches(A, B, match, lower_bound):
            tf_idf_A, tf_idf_B = self.transform(A['target'], B['target'], self.tokenizer)
            csr_matches = self.similarity(tf_idf_A, tf_idf_B.transpose(), 10, lower_bound)
            match.extend(self.matcher(csr_matches, A['id'], B['id']))
            return match

        match = []

        As = [self.left_seller]
        Bs = [self.right_seller]
        for column in exact_match_columns:
            As, Bs = self._split_by_exact_match(As, Bs, column)
        for column in exact_match_columns_with_nan:
            As, Bs = self._split_by_exact_match_with_nan(As, Bs, column)
        if do_ean_check:
            As, Bs = self._split_by_ean(As, Bs)

        for A, B in zip(As, Bs):
            match = compute_matches(A, B, match, lower_bound)

        match = pd.DataFrame(match, columns=['left_side', 'right_side', 'similarity'])
        mark_matches(match)

    def _split_by_ean(self, As, Bs):

        As_splitted = []
        Bs_splitted = []

        for A, B in zip(As, Bs):
            A_ean = A[A['ean'].notna()]
            A_ean = A_ean.reset_index(drop=True)
            A_noean = A[~A['ean'].notna()]
            A_noean = A_noean.reset_index(drop=True)
            B_noean = B[~B['ean'].notna()]
            B_noean = B_noean.reset_index(drop=True)

            if len(A_ean) > 0 and len(B_noean) > 0:

                # we don't want to match products with different EAN
                # we already matched products with the same EAN so if some product from `left_seller` has an EAN, we have to match it only with products without EAN from `right_seller`
                As_splitted.append(A_ean)
                Bs_splitted.append(B_noean)

            if len(A_noean) > 0 and len(B) > 0:

                # but when product from `left_seller` has no EAN, we can match it with products with and also without EAN from `right_seller`
                As_splitted.append(A_noean)
                Bs_splitted.append(B)

        return As_splitted, Bs_splitted

    def _split_by_exact_match(self, As, Bs, column):

        As_splitted = []
        Bs_splitted = []

        for A, B in zip(As, Bs):

            column_values = A[column].value_counts().index
            for column_value in column_values:

                A_sub = A[A[column] == column_value]
                A_sub = A_sub.reset_index(drop=True)

                B_sub = B[B[column] == column_value]
                B_sub = B_sub.reset_index(drop=True)

                if len(A_sub) == 0 or len(B_sub) == 0:
                    continue
                else:
                    As_splitted.append(A_sub)
                    Bs_splitted.append(B_sub)

        return As_splitted, Bs_splitted

    def _split_by_exact_match_with_nan(self, As, Bs, column):
            
            As_splitted = []
            Bs_splitted = []
    
            for A, B in zip(As, Bs):
    
                column_values = A[column].value_counts().index
                for column_value in column_values:
    
                    A_sub = A[(A[column] == column_value) | (A[column].isna())]
                    A_sub = A_sub.reset_index(drop=True)
    
                    B_sub = B[(B[column] == column_value) | (B[column].isna())]
                    B_sub = B_sub.reset_index(drop=True)
    
                    if len(A_sub) == 0 or len(B_sub) == 0:
                        continue
                    else:
                        As_splitted.append(A_sub)
                        Bs_splitted.append(B_sub)
    
            return As_splitted, Bs_splitted

    
    def _drop_matched(self) -> pd.DataFrame:

        def drop_matched_ids(ids, seller_data):
            seller_data.drop(seller_data.loc[seller_data['id'].isin(ids)].index, inplace=True)
            seller_data.reset_index(drop=True, inplace=True)
            return seller_data

        matched_ids = self.matches[~self.matches[self.right_seller_name].isna()][self.left_seller_name]
        self.left_seller = drop_matched_ids(matched_ids, self.left_seller)
        self.right_seller = drop_matched_ids(matched_ids, self.right_seller)


    def _validate_data(self) -> None:
        # TODO: validate that product data have required columns
        pass

    def _make_matches_structure(self) -> None:
        """Creates structure for holding best matches and their similarities.

        Represented as dictionary of pd.DataFrame. Keys are names of sellers taken from keys of self.sellers dictionary.
        Each pd.DataFrame has these columns: ['main_seller_name', 'right_seller_name', 'sim'].
        Values in 'main_seller_name' column are unique identifiers of main seller.
        Values in 'right_seller_name' column are np.nan. Identifiers of best matching right seller will be added here.
        Values in 'sim' column are zeros. Similarity scores of best matching right seller will be added here.
        """

        ids = self.left_seller['id'].values

        self.matches = pd.DataFrame(
            np.array([
                ids, 
                np.empty(len(ids)), 
                np.zeros(len(ids))]).T, 
            columns=[self.left_seller_name, self.right_seller_name, 'sim'])
        self.matches[self.right_seller_name][:] = np.nan 



    # Setters for string similarity model components
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def set_transform(self, transform):
        self.transform = transform

    def set_similarity(self, similarity):
        self.similarity = similarity

    def set_matcher(self, matcher):
        self.matcher = matcher


class StringModel:
    """Wrapper for components that do functionality of a matching model.
    """

    def __init__(self, tokenizer=None, transform=None, similarity=None, matcher=None) -> None:
        """Initializing components of a model.

        Args:
            tokenizer (Callable): a function that takes a string as an input, and splits it into list of string tokens. Default: model.tokenizer.words
            transform (Callable): a function that takes two pd.Series of strings and tokenizer as inputs, and both pd.Series converts into sparse matrices.
                Default: model.transform.tfidf
            similarity (Callable): a function that takes two sparse matrices (and some additional parameters), 
                and computes similarity between all rows from first matrix and all rows from second matrix. 
                Results are also returned as sparse matrix.
                Default: model.similarity.cossim_top
            matcher (Callable): a function that takes sparse matrix of similarities and two pd.Series of ids of input data,
                and returns best matching pairs and their matching score.
                Default: model.matcher.best_unique
        """

        self.tokenizer = words if tokenizer is None else tokenizer
        self.transform = tfidf if transform is None else transform
        self.similarity = cossim_top if similarity is None else similarity
        self.matcher = best_unique if matcher is None else matcher

    def fit(self, A:pd.DataFrame, B:pd.DataFrame) -> None:
        """Transforming input data into numerical vectors.

        Args:
            A (pd.DataFrame(columns=['target', 'id'])): pd.DataFrame for products from first seller. Each row is one product. 
                Columns 'id' is unique identification of a product and column 'target' contains text information about product that will be used for matching.
            B (d.DataFrame(columns=['target', 'id'])): pd.DataFrame for products from second seller. Meaning is the same as for A.
        """
        self.A = A
        self.B = B
        self.tf_idf_A, self.tf_idf_B = self.transform(A['target'], B['target'], self.tokenizer)

    def predict(self, lower_bound=0.5):
        """Computes the best matching pairs of products.

        Args:
            lower_bound (float, optional): Only pairs with similarity higher then lower_bound are kept for further processing. Defaults to 0.5.

        Returns:
            (the same as return type of 'matcher'): the best pairs of products and their matching score
        """
        # note: do not forget to transpose the second matrix
        csr_matches = self.similarity(self.tf_idf_A, self.tf_idf_B.transpose(), 10, lower_bound)
        return self.matcher(csr_matches, self.A['id'], self.B['id'])