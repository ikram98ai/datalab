import pandas as pd

def proc_data(df):
    # Remove object columns having more than 10 unique values
    object_columns = df.select_dtypes(include='object').columns
    columns_to_drop = [col for col in object_columns if df[col].nunique() > 10]
    df.drop(columns_to_drop, axis=1, inplace=True)
    # Remove 50% NaN values columns
    nan_percent = df.isnull().mean() * 100
    valid_columns = nan_percent[nan_percent <= 50].index.tolist()
    df = df[valid_columns]
    # Fill_na with modes of columns
    modes = df.mode().iloc[0]
    df = df.fillna(modes)
    # Transform columns to onehot encoding
    object_columns = object_columns.difference(columns_to_drop)
    df_encoded = pd.get_dummies(df[object_columns], drop_first=True)
    df = pd.concat([df.drop(object_columns, axis=1), df_encoded], axis=1)

    return df
