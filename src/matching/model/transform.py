from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Callable

def tfidf(A:pd.Series, B:pd.Series, tokenizer:Callable) -> tuple[csr_matrix, csr_matrix]:
    """Splits string into list of tokens using 'tokenizer' and transforms list of tokens into numerical vector using Tf-Idf algorithm.
    
    Done for each item of pd.Series, so the result is matrix. And since we might work with big amount of data, result is a sparse matrix.

    Args:
        A (pd.Series): pd.Series of strings for products from first seller. Each row is one product. 
        B (pd.Series): pd.Series of strings for products from second seller. Each row is one product. 
        tokenizer (Callable): a function that takes a string as an input, and splits it into list of string tokens

    Returns:
        tuple(csr_matrix, csr_matrix): sparse matrice representation of input pd.Series
    """

    vectorizer = TfidfVectorizer(min_df=1, analyzer=tokenizer)
    tf_idf_A = vectorizer.fit_transform(A)
    tf_idf_B = vectorizer.transform(B)

    return tf_idf_A, tf_idf_B