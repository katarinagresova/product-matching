import pandas as pd

def best_unique(sparse_matrix, A, B) -> list:
    """[summary]
    Let's say our sparse matrix looks like this:
    [[1, 2, 0]
     [2, 0, 3]
     [4, 0, 5]].
    We want only one match for each source seller, so we will keep only the highest value in each row:
    [[0, 2, 0]
     [0, 0, 3]
     [0, 0, 5]].
    And we want unique matches, so we also keep only the highest value in each column:
    [[0, 2, 0]
     [0, 0, 0]
     [0, 0, 5]].
    Non-zero entries are our best unique matches. If some row contains only zero entries, it means that we
    haven't found a match for that row.

    Args:
        sparse_matrix ([type]): [description]
        A ([type]): [description]
        B ([type]): [description]

    Returns:
        list: [description]
    """
    
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    array = [[sparserows[i], sparsecols[i], sparse_matrix.data[i]] for i in range(len(sparse_matrix.data))]
    df = pd.DataFrame(array, columns=['row', 'col', 'sim'])

    # sort the whole dataframe by similarity values - starting from highest similarity
    df.sort_values(by=['sim'], ascending=False, inplace=True, ignore_index=True)

    # we want only one match for each source seller - for each product, keep only results with the highest similarity
    ind_list = df['row'].drop_duplicates(keep="first").index
    df = df.iloc[ind_list]
    df.reset_index(inplace=True, drop=True)

    # we want unique matches - if some target product is duplicit, keep only the one with the highest similarity score
    ind_list = df['col'].drop_duplicates(keep="first").index
    df = df.iloc[ind_list]
    df.reset_index(inplace=True, drop=True)

    if len(df) > 0:
      return list(df.apply(lambda x: [A[x['row']], B[x['col']], x['sim']], axis=1))
    else:
      return []