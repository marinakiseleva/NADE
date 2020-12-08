"""

Run given dataset  with NADEs to compute likelihood

run using:
        python2 run_thex.py

"""

import sys
if not (sys.version_info.major == 2 and sys.version_info.minor == 7):
    raise ValueError("Must run this on Python 2.7")


#  Imports and paths updates
import os
test_path = "/Users/marina/Documents/PhD/research/astro_research/data/testing/"
data_path = test_path + "PROCESSED_DATA/"
os.environ["DATASETSPATH"] = data_path
os.environ["RESULTSPATH"] = "./output"
os.environ["PYTHONPATH"] = "./buml:$PYTHONPATH"
sys.path
sys.path.append('buml')

import copy
from optparse import OptionParser
import pandas as pd
import numpy as np
from collections import OrderedDict
from originalNADE import *
import Data.utils


# Default settings for all NADEs used
NADE_CONSTS = ["--theano",
               "--form", "MoG",
               "--dataset", "HDF5_FILE_NAME_HERE",
               "--training_route", "/folds/1/training/(1|2|3|4|5|6|7|8)",
               "--validation_route", "/folds/1/training/9",
               "--test_route", "/folds/1/tests/.*",
               "--samples_name", "data",
               "--hlayers", "2",  # 2 hidden layers
               "--layerwise",
               "--lr", "0.0005",  # learning rate
               "--wd", "0.002",  # weight decay not using...... pretty sure
               "--n_components", "10",  # number of GMM components
               "--epoch_size", "100",
               "--momentum", "0.9",
               "--units", "100",  # units in hidden layer (I think)
               # "--pretraining_epochs", "5",
               # "--validation_loops", "20", # for orderless NADE
               "--epochs", "20",  # maximum number of epochs
               # "--normalize", "False",
               "--batch_size", "32",
               "--show_training_stop", "True"]
# Other params
# Nonlinearity function defaults to ReLU


def get_class_nade(class_name, dataset_name):
    parser = get_parser()
    # Update HDF5_FILE_NAME_HERE for this class
    class_args = copy.deepcopy(NADE_CONSTS)
    del class_args[4]
    class_args.insert(4, class_name.replace(" ", "_") + "X.hdf5")
    options, args = parser.parse_args(class_args)
    nade = train_NADE(options, [dataset_name])

    if options.show_training_stop:
        print("\n\n Training Stop Print Out for    " + class_name)
        dataset_file = data_path + options.dataset
        training_dataset = Data.BigDataset(
            dataset_file, options.training_route, options.samples_name)
        testing_dataset = Data.BigDataset(
            dataset_file, options.test_route, options.samples_name)

        training_likelihood = nade.estimate_loglikelihood_for_dataset(
            training_dataset)
        testing_likelihood = nade.estimate_loglikelihood_for_dataset(
            testing_dataset)
        print("Training log likelihood " + str(training_likelihood))
        print("Testing log likelihood " + str(testing_likelihood))
        print("\n\n")

    return nade


def get_nlls(col_sample, class_nades):
    """
    Get likelihood of each class for this row
    Return as dict.
    :param col_sample: Sample being evaluated as column vector
    :param class_nades: Map from class name to trained NADE
    """
    class_nll = OrderedDict()
    for class_name in class_nades.keys():
        cnade = class_nades[class_name]
        log_density = cnade.logdensity(col_sample)[0]
        class_nll[class_name] = log_density
    return class_nll


def get_metrics(lls, y):

    class_names = list(lls)

    label_col = list(y)[0]
    print(label_col)

    # Map from class name to TP, FP, FN, and TN rates in dict.
    class_mets = {cn: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cn in class_names}

    for class_name in class_names:
        for index, ll in lls.iterrows():
            label = str(y.iloc[index][label_col])
            values = ll.tolist()
            max_class_index = values.index(max(values))
            pred_class = class_names[max_class_index]

            # this class is the label, and we predicted it as max
            if label == class_name and pred_class == label:
                class_mets[class_name]["TP"] += 1

            # this class is not the label, but it was the max likelihood
            elif label != class_name and pred_class == class_name:
                class_mets[class_name]["FP"] += 1

            # this class is not the label, and its not the max (TN)
            elif label != class_name and pred_class != class_name:
                class_mets[class_name]["TN"] += 1

            # this class is label, but its not the max (FN)
            elif label == class_name and pred_class != class_name:
                class_mets[class_name]["FN"] += 1

    return class_mets


def get_performance(lls, y, classes):
    class_mets = get_metrics(lls, y)

    # Purity TP / (TP+FP)
    purities = OrderedDict()
    for cname in classes:
        m = class_mets[cname]
        p = m["TP"] / (m["TP"] + m["FP"] + 0.0)
        purities[cname] = p
        print(cname + ' purity : ' + str(round(p * 100, 1)) + "%")

    total_TPs = 0
    for cname in classes:
        total_TPs += class_mets[cname]["TP"]
    acc = total_TPs / (y.shape[0] + 0.0)
    print('Accuracy : ' + str(round(acc * 100, 1)) + "%")

    return purities, acc


def get_pred_class(lls):
    """
    Get MAP - class with max likelihood assigned
    Gets max key of dict passed in
    :param lls: Map from class name to log likelihood
    """
    return max(lls, key=lls.get)


def run_model(dataset_name, classes):
    class_nades = OrderedDict()
    for class_name in classes:
        print("\nTraining NADE for class " + str(class_name))
        class_nades[class_name] = get_class_nade(class_name, dataset_name)

    X_test_file = "test_X.hdf5"  # HDF5
    y_test_file = "test_y.csv"  # CSV

    test_X_hdf5 = Data.BigDataset(
        data_path + X_test_file,
        "/folds/1/tests/.*",
        "data")

    test_y = pd.read_csv(data_path + y_test_file)
    test_X = test_X_hdf5.get_file(0, 0)
    num_rows = test_X.shape[0]

    if num_rows != test_y.shape[0]:
        raise ValueError("Bad.")

    # Save all assigned likelihoods for test data
    saved_loglikelihoods = []
    for i in range(num_rows):
        row = test_X[i]
        col_sample = np.atleast_2d(row).T  # transpose of row
        lls = []
        for class_name in classes:
            cnade = class_nades[class_name]
            ll = cnade.logdensity(col_sample)[0]
            lls.append(ll)
        saved_loglikelihoods.append(lls)

    lls_df = pd.DataFrame(saved_loglikelihoods, columns=classes)
    lls_df.to_csv(test_path + "OUTPUT/" + "loglikelihoods.csv", index=False)

    purities, acc = get_performance(lls_df, test_y, classes)

    return purities, acc


def get_mean_stdev(l):
    """
    Get mean and standard deviation of list
    """
    arr = np.array(l)
    mean = np.average(arr)
    mean = str(round(mean * 100, 1)) + "%"

    stdev = np.std(arr)
    stdev = str(round(stdev * 100, 1))
    return mean, stdev

if __name__ == "__main__":
    """
    Create NADE per class.
    """
    ########################################################
    # SET CLASSES MANUALLY HERE.
    ########################################################

    # THEx Data
    # classes = ['Unspecified Ia', 'Unspecified II']
    # classes = ['Unspecified Ia', 'Unspecified Ia Pec', 'Ia-91T', 'Ia-91bg', 'Ib/c', 'Unspecified Ib', 'IIb', 'Unspecified Ic', 'Ic Pec', 'Unspecified II', 'II P', 'IIn', 'TDE', 'GRB']
    # classes = ['TDE', 'Unspecified Ia', 'Unspecified II']

    # SYNTHETIC DATA TEST
    # classes = ['dog', 'cat', 'mouse']

    # WINE DATASET
    classes = ["4", "5", "6", "7"]
    dataset_name = "wine"
    runs = 10

    ########################################################
    all_ps = []
    all_accs = []
    for i in range(runs):
        purities, acc = run_model(dataset_name, classes)
        all_ps.append(purities)
        all_accs.append(acc)

    avg, stdev = get_mean_stdev(all_accs)
    print("Accuracy " + avg + u"\u00B1" + stdev)

    for class_name in classes:
        class_p = []
        for p_map in all_ps:
            class_p.append(p_map[class_name])
        avg, stdev = get_mean_stdev(class_p)
        print("Purity " + class_name + " :" + avg + u"\u00B1" + stdev)
