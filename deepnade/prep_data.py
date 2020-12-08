import os
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict


test_path = "/Users/marina/Documents/PhD/research/astro_research/data/testing/"
dpath = test_path + "PROCESSED_DATA/"


def prettify(class_name):
    if "/" in class_name:
        class_name = class_name.replace("/", "")
    class_name = class_name.replace(" ", "_")
    return class_name


def save_HDF5s(training_folds, val_fold, test_fold, thex_data_path):
    """
    Save class data to HDF5
    """
    # Save to HDF5 File
    hfile = h5py.File(thex_data_path, 'w')
    # define & fill groups
    for i in range(8):
        training = hfile.create_group("folds/1/training/" + str(i + 1))
        data = training_folds[i].to_numpy(dtype=np.float32)
        dset = training.create_dataset("data", data=data)
    val = hfile.create_group("folds/1/training/9")
    dset = val.create_dataset("data", data=val_fold.to_numpy(dtype=np.float32))
    val = hfile.create_group("folds/1/tests/1")
    dset = val.create_dataset("data", data=test_fold.to_numpy(dtype=np.float32))
    hfile.close()


def save_CSVs(fold_sets, class_X, class_name, output_dir):
    """
    Save class data to CSV
    """
    train_indices = []
    for i in range(9):  # Include validation fold in training
        train_indices += fold_sets[i].tolist()
    class_train = class_X.iloc[train_indices]
    class_test = class_X.iloc[fold_sets[9]]
    class_train.to_csv(output_dir + prettify(class_name) + "train.csv", index=False)
    class_test.to_csv(output_dir + prettify(class_name) + "test.csv", index=False)


def save_class_data(class_name, X, y, output_dir, scaling=False):
    """
    Save the X data of this class as HDF5 file
    Returns test fold to be saved separately in joined test file.
    """
    label_name = list(y)[0]
    class_indices = y.loc[y[label_name].str.contains(class_name)].index
    class_X = X.iloc[class_indices]
    # Divide data into 10 folds; use 8 as training, 1 as validation, 1 as testing
    kf = KFold(n_splits=10, shuffle=True)
    fold_sets = []
    for remaining_indices, fold_indices in kf.split(class_X):
        fold_sets.append(fold_indices)
    training_folds = []
    for i in range(8):
        training_folds.append(class_X.iloc[fold_sets[i]])

    val_fold = class_X.iloc[fold_sets[8]]
    test_fold = class_X.iloc[fold_sets[9]]

    if scaling:
        fs = list(val_fold)
        scaler = StandardScaler()
        class_X = pd.DataFrame(
            data=scaler.fit_transform(class_X),
            columns=fs)

        training_folds = []
        for i in range(8):
            training_folds.append(class_X.iloc[fold_sets[i]])

        val_fold = pd.DataFrame(
            data=scaler.transform(val_fold), columns=fs)
        test_fold = pd.DataFrame(
            data=scaler.transform(test_fold), columns=fs)

    # Save to HDF5 File
    class_path = output_dir + prettify(class_name) + 'X.hdf5'
    save_HDF5s(training_folds, val_fold, test_fold, class_path)

    # Also save as CSVs - to test on KDE Model
    save_CSVs(fold_sets, class_X, class_name, output_dir)

    return test_fold


def save_test_X_y(test_folds, dpath, label="transient_type"):
    """
    Using existing folds, combine each class's test fold into one whole test data set.
    Save as both an HDF5 and CSV. 
    """
    full_test_set = pd.concat(test_folds.values())
    hfile = h5py.File(dpath + "test_X.hdf5", 'w')
    group = hfile.create_group("folds/1/tests/1")
    dset = group.create_dataset("data", data=full_test_set.to_numpy(dtype=np.float32))
    hfile.close()
    # Save as CSV too for KDE model testing
    full_test_set.to_csv(dpath + "test_X.csv", index=False)

    # Save labels corresponding to test set in CSV.
    labels = []
    for class_name in test_folds.keys():
        class_count = test_folds[class_name].shape[0]
        for i in range(class_count):
            labels.append(class_name)

    label_df = pd.DataFrame(labels, columns=[label])
    label_df.to_csv(dpath + "test_y.csv", index=False)


def save_train_X_y(dpath, classes, label="transient_type"):
    """
    Load train X CSV files, and combine all into one X training file
    Save labels in correct order in corresponding y file.
    For use in KDE model.
    """
    class_dfs = []
    labels = []
    for class_name in classes:
        class_X = pd.read_csv(dpath + prettify(class_name) + "train.csv")
        class_dfs.append(class_X)
        for i in range(class_X.shape[0]):
            labels.append(class_name)
    X = pd.concat(class_dfs)
    X.to_csv(dpath + "train_X.csv", index=False)
    y = pd.DataFrame(labels, columns=[label])
    y.to_csv(dpath + "train_y.csv", index=False)
