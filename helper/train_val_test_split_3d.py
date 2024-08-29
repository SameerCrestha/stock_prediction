import numpy as np

def train_val_test_split_3d(array_3d, train_ratio=0.7, val_ratio=0.15, shuffle=False):
    """
    Splits a 3D NumPy array into train, validation, and test sets.

    Parameters:
    array_3d (np.ndarray): The input 3D array of shape (num_samples, window_size, num_features).
    train_ratio (float): The ratio of data to be used for training (default is 0.7).
    val_ratio (float): The ratio of data to be used for validation (default is 0.15).
    shuffle (bool): Whether to shuffle the data before splitting (default is False).

    Returns:
    tuple: A tuple containing the train, validation, and test sets as 3D arrays.
    """
    # Ensure that the sum of train and val ratios is less than 1
    assert train_ratio + val_ratio <= 1, "Sum of train_ratio and val_ratio must be <= 1."
    
    # Calculate the implicit test ratio
    test_ratio = 1 - (train_ratio + val_ratio)
    
    # Get the number of samples
    num_samples = array_3d.shape[0]
    
    # Optionally shuffle the data
    if shuffle:
        np.random.shuffle(array_3d)
    
    # Calculate split indices
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)
    
    # Split the data
    train_set = array_3d[:train_end]
    val_set = array_3d[train_end:val_end]
    test_set = array_3d[val_end:]
    
    return train_set, val_set, test_set
