import pandas as pd
import numpy as np

def make_pairing_dataset(data, id_columns, match_columns, not_match_columns):

    # rename column url to id
    data = data.rename(columns={'url': 'id'})

    # map NaN to ''
    data = data.fillna('')

    # make group_id
    data['group_id'] = data.apply(lambda x: ' '.join(str(x[column]).lower() for column in id_columns), axis=1)

    # make target
    target_columns = [column for column in data.columns if column not in not_match_columns and column not in match_columns and column not in id_columns and column not in ['group_id', 'target', 'id', 'ean']]
    data['target'] = data.apply(lambda x: ' '.join([x[column].lower() for column in target_columns]), axis=1)

    # keep only needed columns
    final_columns = ['id', 'ean', 'group_id', 'target'] + match_columns
    return data[final_columns]

def make_classification_dataset(client, sellers, real_pairs, excluded_columns):

    # compute intersection of columns
    columns = list(client['data'].columns.values)
    for seller in sellers:
        columns.extend(seller['data'].columns.values)
    columns = list(set(i for i in columns if columns.count(i) == len(sellers) + 1))
    columns = [item for item in columns if item not in excluded_columns]
    print("Using columns:", columns)

    class_pairs = []
    client_df = client['data']
    for i in range(len(real_pairs)):

        for seller in sellers:
            seller_df = seller['data']

            # other seller is selling this product as well (there is not an 'NaN' in cell)
            if not pd.isna(real_pairs.iloc[i][seller['name']]):

                row = [1]
                row.extend(client_df.loc[client_df['url'] == real_pairs.iloc[i][client['name']]][columns].values[0])
                row.extend(seller_df.loc[seller_df['url'] == real_pairs.iloc[i][seller['name']]][columns].values[0])

                class_pairs.append(row)

            found = False
            while not found:

                random_non_match = np.random.randint(0, len(real_pairs))

                if (random_non_match != i) and (not pd.isna(real_pairs.iloc[random_non_match][seller['name']])):

                    row = [0]
                    row.extend(client_df.loc[client_df['url'] == real_pairs.iloc[i][client['name']]][columns].values[0])
                    row.extend(seller_df.loc[seller_df['url'] == real_pairs.iloc[random_non_match][seller['name']]][columns].values[0])

                    class_pairs.append(row)
                    found = True

    pairs_columns = ['label'] + ['left_' + header for header in columns] + ['right_' + header for header in columns]
    class_pairs_df = pd.DataFrame(class_pairs, columns=pairs_columns)
    print("Labels distribution:")
    print(class_pairs_df['label'].value_counts())

    return class_pairs_df