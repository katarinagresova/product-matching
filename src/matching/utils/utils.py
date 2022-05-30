from sklearn.model_selection import train_test_split

def split_train_val_test(data, train_ratio = 0.7, validation_ratio = 0.2, test_ratio = 0.1):

    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 0.00000001:
        raise ValueError("Sum of all ratios must be equal to one.")

    # Produces test split.
    x_remaining, x_test = train_test_split(data, test_size=test_ratio, random_state=42)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - test_ratio
    ratio_val_adjusted = validation_ratio / ratio_remaining

    # Produces train and val splits.
    x_train, x_val = train_test_split(x_remaining, test_size=ratio_val_adjusted, random_state=42)

    return x_train, x_val, x_test