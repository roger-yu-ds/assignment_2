import pandas as pd
from pathlib import Path, WindowsPath
from dotenv import find_dotenv
from typing import Tuple

project_dir = Path(find_dotenv()).parent

def save_sets(X_train: pd.DataFrame = None,
              y_train: pd.Series = None,
              X_val: pd.DataFrame = None,
              y_val: pd.Series = None,
              X_test: pd.DataFrame = None,
              y_test: pd.Series = None,
              path: WindowsPath = None):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    if X_train is not None:
        X_train.to_csv(path.joinpath('X_train.csv'), index=False)
    if y_train is not None:
        y_train.to_csv(path.joinpath('y_train.csv'), index=False)
    if X_val is not None:
        X_val.to_csv(path.joinpath('X_val.csv'), index=False)
    if y_val is not None:
        y_val.to_csv(path.joinpath('y_val.csv'), index=False)
    if X_test is not None:
        X_test.to_csv(path.joinpath('X_test.csv'), index=False)
    if y_test is not None:
        y_test.to_csv(path.joinpath('y_test.csv'), index=False)


def load_sets(path: WindowsPath = project_dir / 'data/processed') -> \
        Tuple[pd.DataFrame,
              pd.DataFrame,
              pd.Series,
              pd.Series]:
    """

    :param path:
    :return:
    """
    X_train = pd.read_csv(path.joinpath('X_train').with_suffix('.csv'))
    X_test = pd.read_csv(path.joinpath('X_test').with_suffix('.csv'))
    y_train = pd.read_csv(path.joinpath('y_train').with_suffix('.csv')).squeeze()
    y_test = pd.read_csv(path.joinpath('y_test').with_suffix('.csv')).squeeze()

    return X_train, X_test, y_train, y_test

def subset_x_y(target, features, start_index: int, end_index: int):
    """Keep only the rows for X and y sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """

    return features[start_index:end_index], target[start_index:end_index]


def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(target) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy,
                                  start_index=0, end_index=-cutoff * 2)
    X_val, y_val = subset_x_y(target=target, features=df_copy,
                              start_index=-cutoff * 2, end_index=-cutoff)
    X_test, y_test = subset_x_y(target=target, features=df_copy,
                                start_index=-cutoff, end_index=len(target))

    return X_train, y_train, X_val, y_val, X_test, y_test


