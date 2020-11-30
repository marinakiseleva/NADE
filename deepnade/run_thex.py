"""

Run THEx model with NADEs to compute likelihood

run using:
        python2 run_thex.py

"""

#  Imports and paths updates
import os
data_path = "/Users/marina/Documents/PhD/research/astro_research/data/testing/"
os.environ["DATASETSPATH"] = data_path
os.environ["RESULTSPATH"] = "./output"
os.environ["PYTHONPATH"] = "./buml:$PYTHONPATH"
import sys
sys.path
sys.path.append('buml')
import copy
from optparse import OptionParser
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
               "--lr", "0.02",  # learning rate
               "--wd", "0.02",  # weight decay
               "--n_components", "10",  # number of GMM components
               "--epoch_size", "100",
               "--momentum", "0.9",
               "--units", "100",  # units in hidden layer (I think)
               "--pretraining_epochs", "5",
               "--validation_loops", "20",
               "--epochs", "20",  # number of epochs
               "--normalize",
               "--batch_size", "100",
               "--show_training_stop", "True"]


def get_class_nade(class_name):
    parser = get_parser()
    # Update HDF5_FILE_NAME_HERE for this class
    class_args = copy.deepcopy(NADE_CONSTS)
    del class_args[4]
    class_args.insert(4, class_name.replace(" ", "_") + "X.hdf5")
    options, args = parser.parse_args(class_args)
    nade = train_NADE(options, args)
    return nade


if __name__ == "__main__":
    """
    Create NADE per class.
    """
    nade = get_class_nade("Unspecified Ia")

    # dataset_file = data_path + options.dataset
    # test_dataset = Data.BigDataset(
    #     dataset_file, options.test_route, options.samples_name)

    # # Sample and see likelihood of one sample
    # print("\n\n Consider sample likelihood from test set. ")
    # sample = test_dataset.get_file(0, 0)[1]

    # col_sample = np.atleast_2d(sample).T
    # ld = nade.logdensity(col_sample)
    # print("Log density " + str(ld))
