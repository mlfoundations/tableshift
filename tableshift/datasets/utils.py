import pandas as pd


def convert_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Utility function for automatically casting int-valued columns to float."""
    for c in df.columns:
        df[c] = df[c].convert_dtypes(convert_string=False,
                                     convert_boolean=False)
    return df


def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how='any')


def apply_column_missingness_threshold(df: pd.DataFrame,
                                       missingness_threshold=0.8) -> pd.DataFrame:
    miss = pd.isnull(df).sum() / len(df)

    dropcols = miss.index[miss >= missingness_threshold].tolist()
    df.drop(columns=dropcols, inplace=True)
    return df
