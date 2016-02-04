#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import re
import pickle
import argparse
import numpy as np
from os.path import join as pjoin

from smartlearner import views
from smartlearner.status import Status

from convnade import utils
from convnade import datasets
from convnade.utils import Timer

from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask

DATASETS = ['binarized_mnist']


def build_argparser():
    DESCRIPTION = "Evaluate the NLL estimate of a ConvNADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--seed', type=int,
                   help="Seed used to choose the orderings. Default: 1234", default=1234)

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')

    return p

def estimate_NLL(model, dataset, seed=1234):
    loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, dataset)
    status = Status()

    batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(dataset,
                                                               batch_size=len(dataset),
                                                               concatenate_mask=model.nb_channels == 2,
                                                               keep_mask=True,
                                                               seed=seed)

    nll = views.LossView(loss=loss, batch_scheduler=batch_scheduler)

    # Try different size of batch size.
    batch_size = len(dataset)
    while batch_size >= 1:
        print("Estimating NLL using batch size of {}".format(batch_size))
        try:
            batch_scheduler.batch_size = batch_size
            return float(nll.mean.view(status)), float(nll.stderror.view(status))

        except MemoryError as e:
            # Probably not enough memory on GPU
            #print("\n".join([l for l in str(e).split("\n") if "allocating" in l]))
            pass

        print("*An error occured while estimating NLL. Will try a smaller batch size.")
        batch_size = batch_size // 2

    raise RuntimeError("Cannot find a suitable batch size to estimate NLL. Try using CPU instead or a GPU with more memory.")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    if not os.path.isdir(pjoin(experiment_path, "model")):
        parser.error('Cannot find model for experiment: {0}!'.format(experiment_path))

    if not os.path.isfile(pjoin(experiment_path, "hyperparams.json")):
        parser.error('Cannot find hyperparams for experiment: {0}!'.format(experiment_path))

    # Load experiments hyperparameters
    hyperparams = utils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))

    with Timer("Loading dataset"):
        trainset, validset, testset = datasets.load(hyperparams['dataset'], keep_on_cpu=True)
        print(" (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)), end="")

    with Timer("Loading model"):
        if hyperparams["model"] == "convnade":
            from convnade import DeepConvNADE
            model_class = DeepConvNADE

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path, "model"))  # Create new instance
        model.load(pjoin(experiment_path, "model"))  # Restore state.

        print(str(model.convnet))
        print(str(model.fullnet))

    # Result files.
    result_file = pjoin(experiment_path, "results_estimate.json")

    if not os.path.isfile(result_file) or args.force:
        with Timer("Evaluating NLL estimate"):
            results = {}
            results['NLL_est._trainset'] = estimate_NLL(model, trainset, seed=args.seed)
            results['NLL_est._validset'] = estimate_NLL(model, validset, seed=args.seed)
            results['NLL_est._testset'] = estimate_NLL(model, testset, seed=args.seed)
            utils.save_dict_to_json_file(result_file, results)
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = utils.load_dict_from_json_file(result_file)

    print("NLL estimate on trainset: {:.2f} ± {:.2f}".format(*results['NLL_est._trainset']))
    print("NLL estimate on validset: {:.2f} ± {:.2f}".format(*results['NLL_est._validset']))
    print("NLL estimate on testset:  {:.2f} ± {:.2f}".format(*results['NLL_est._testset']))

if __name__ == '__main__':
    main()
