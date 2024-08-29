def concatenated_windows_to_3d_array(df, window_size):
    """
    Converts a concatenated rolling window DataFrame into a 3D NumPy array.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame which is a concatenation of rolling windows.
    window_size (int): The size of each window in the concatenated DataFrame.
    
    Returns:
    np.ndarray: A 3D NumPy array of shape (num_windows, window_size, num_features).
    """
    # Convert DataFrame to NumPy array
    data = df.to_numpy()
    
    # Calculate the number of windows
    num_windows = len(data) // window_size
    num_features = data.shape[1]
    
    # Reshape the data into a 3D array
    rolling_array = data.reshape((num_windows, window_size, num_features))
    
    return rolling_array