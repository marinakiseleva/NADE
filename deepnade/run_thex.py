"""

Run THEx model with NADEs to compute likelihood

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
               "--lr", "0.002",  # learning rate
               "--wd", "0.002",  # weight decay
               "--n_components", "10",  # number of GMM components
               "--epoch_size", "100",
               "--momentum", "0.9",
               "--units", "100",  # units in hidden layer (I think)
               # "--pretraining_epochs", "5",
               # "--validation_loops", "20", # for orderless NADE
               "--epochs", "20",  # maximum number of epochs
               # "--normalize", "False",
               "--batch_size", "16",
               "--show_training_stop", "True"]
# Other params
# Nonlinearity function defaults to ReLU


def get_class_nade(class_name):
    parser = get_parser()
    # Update HDF5_FILE_NAME_HERE for this class
    class_args = copy.deepcopy(NADE_CONSTS)
    del class_args[4]
    class_args.insert(4, class_name.replace(" ", "_") + "X.hdf5")
    options, args = parser.parse_args(class_args)
    nade = train_NADE(options, ["THEx"])

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


def get_pred_class(lls):
    """ 

    Get MAP - class with max likelihood assigned
    Gets max key of dict passed in
    :param lls: Map from class name to log likelihood
    """

    return max(lls, key=lls.get)

if __name__ == "__main__":
    """
    Create NADE per class.
    """
    # classes = ['Unspecified Ia', 'Unspecified Ia Pec', 'Ia-91T', 'Ia-91bg', 'Ib/c', 'Unspecified Ib', 'IIb', 'Unspecified Ic', 'Ic Pec', 'Unspecified II', 'II P', 'IIn', 'TDE', 'GRB']
    # classes = ['TDE', 'Unspecified Ia', 'Unspecified II']

    ########################################################
    # SET CLASSES MANUALLY HERE.
    ########################################################

    # THEx Data
    # classes = ['Unspecified Ia', 'Unspecified II']

    # SYNTHETIC DATA TEST
    classes = ['dog', 'cat', 'mouse']

    ########################################################

    class_nades = OrderedDict()
    for class_name in classes:
        print("\nTraining NADE for class " + str(class_name))
        class_nades[class_name] = get_class_nade(class_name)

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

    output = pd.DataFrame(saved_loglikelihoods, columns=classes)
    output.to_csv(test_path + "OUTPUT/" + "loglikelihoods.csv", index=False)
