import pandas as pd

def rolling_window(df, window_size):
    """
    Generates a rolling window list of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The size of the rolling window.

    Returns:
        list: A list of DataFrames, each representing a window of the input DataFrame.
    """

    if window_size <= 0:
        raise ValueError("Window size must be positive")

    if window_size > len(df):
        raise ValueError("Window size cannot be larger than the DataFrame length")

    rolling_windows = []
    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i:i+window_size].copy()
        rolling_windows.append(window_df)

    return rolling_windows