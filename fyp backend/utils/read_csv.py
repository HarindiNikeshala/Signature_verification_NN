# import pandas as pd
# import numpy as np
# import keras
# n_input = 9

# def readCSV(train_path, test_path, type2=False):
#     # Reading train data
#     df = pd.read_csv(train_path, usecols=range(n_input))
#     train_input = np.array(df.values)
#     train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
#     df = pd.read_csv(train_path, usecols=(n_input,))
#     temp = [elem[0] for elem in df.values]
#     correct = np.array(temp)
#     corr_train = keras.utils.to_categorical(correct,2)      # Converting to one hot
#     # Reading test data
#     df = pd.read_csv(test_path, usecols=range(n_input))
#     test_input = np.array(df.values)
#     test_input = test_input.astype(np.float32, copy=False)
#     if not(type2):
#         df = pd.read_csv(test_path, usecols=(n_input,))
#         temp = [elem[0] for elem in df.values]
#         correct = np.array(temp)
#         corr_test = keras.utils.to_categorical(correct,2)      # Converting to one hot
#     if not(type2):
#         return train_input, corr_train, test_input, corr_test
#     else:
#         return train_input, corr_train, test_input

import pandas as pd
import numpy as np
import keras

n_input = 9

def readCSV(train_path, test_path, type2=False):
    """
    Reads training and testing CSV files.

    Training CSV is assumed to have a header row; Testing CSV has no header.

    Returns:
      - train_input: np array of shape (num_train_samples, n_input)
      - corr_train: one-hot labels for train data
      - test_input: np array of shape (num_test_samples, n_input)
      - corr_test (if type2=False): one-hot labels for test data
    """
    # --- Training data ---
    # Read with header row (column names), features are first n_input columns
    df_train = pd.read_csv(train_path, usecols=range(n_input), header=0)
    train_input = df_train.values.astype(np.float32, copy=False)

    df_labels_train = pd.read_csv(train_path, usecols=[n_input], header=0)
    labels_train = df_labels_train.values.flatten()
    corr_train = keras.utils.to_categorical(labels_train, 2)

    # --- Testing data ---
    # No header row in test CSV
    df_test = pd.read_csv(test_path, usecols=range(n_input), header=None)
    test_input = df_test.values.astype(np.float32, copy=False)

    if not type2:
        df_labels_test = pd.read_csv(test_path, usecols=[n_input], header=None)
        labels_test = df_labels_test.values.flatten()
        corr_test = keras.utils.to_categorical(labels_test, 2)
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input
