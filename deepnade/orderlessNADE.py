import os
from optparse import OptionParser
import Instrumentation
import Backends
import Optimization
import TrainingController
import numpy as np
import Utils
import Data.utils
import scipy.stats
import gc
from Utils.DropoutMask import create_dropout_masks
from Utils.theano_helpers import floatX


import NADE

# python original_NADE.py --dataset original_NADE/binarized_mnist.hdf5 --epoch_size 1000 --momentum 0.9 --lr 0.001 --deep --hlayers 2 --units 1000 --orderless --no_validation NADE_orderless/2h_1000
# np.seterr(invalid = "raise", divide="raise")


def log_message(backends, message):
    for b in backends:
        b.write([], "", message)


def get_parser():
    parser = OptionParser(usage="usage: %prog [options] results_route")
    parser.add_option("--theano", dest="theano", default=False, action="store_true")
    # Model options
    parser.add_option("--form", dest="form", default="")
    parser.add_option("--n_quantiles", dest="n_quantiles", default=20, type="int")
    parser.add_option("--n_components", dest="n_components", default=10, type="int")
    parser.add_option("--hlayers", dest="hlayers", default=1, type="int")
    parser.add_option("--units", dest="units", default=100, type="int")
    parser.add_option("--nonlinearity", dest="nonlinearity", default="RLU")
    # Training options
    parser.add_option("--layerwise", dest="layerwise",
                      default=False, action="store_true")
    parser.add_option("--training_ll_stop", dest="training_ll_stop",
                      default=np.inf, type="float")
    parser.add_option("--lr", dest="lr", default=0.01, type="float")
    parser.add_option("--decrease_constant", dest="decrease_constant",
                      default=0.1, type="float")
    parser.add_option("--wd", dest="wd", default=0.00, type="float")
    parser.add_option("--momentum", dest="momentum", default=0.9, type="float")
    parser.add_option("--epochs", dest="epochs", default=200, type="int")
    parser.add_option("--pretraining_epochs",
                      dest="pretraining_epochs", default=20, type="int")
    parser.add_option("--epoch_size", dest="epoch_size", default=10, type="int")
    parser.add_option("--batch_size", dest="batch_size", default=100, type="int")
    # Dataset options
    parser.add_option("--dataset", dest="dataset", default="")
    parser.add_option("--training_route", dest="training_route", default="train")
    parser.add_option("--validation_route",
                      dest="validation_route", default="validation")
    parser.add_option("--test_route", dest="test_route", default="test")
    parser.add_option("--samples_name", dest="samples_name", default="data")
    parser.add_option("--normalize", dest="normalize",
                      default=False, action="store_true")
    parser.add_option("--validation_loops", dest="validation_loops",
                      default=16, type="int")
    parser.add_option("--no_validation", dest="no_validation",
                      default=False, action="store_true")
    # Reports
    parser.add_option("--show_training_stop", dest="show_training_stop",
                      default=False, action="store_true")
    parser.add_option("--summary_orderings", dest="summary_orderings",
                      default=10, type="int")
    parser.add_option("--report_mixtures", dest="report_mixtures",
                      default=False, action="store_true")

    gc.set_threshold(gc.get_threshold()[0] / 5)
    # gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES | gc.DEBUG_OBJECTS)
    return parser


def prep_datasets(options, hdf5_backend):
    """
    Using passed-in options, return training, validation, testing, and 'masked' datasets
    """
    # Read datasets
    dataset_file = os.path.join(os.environ["DATASETSPATH"], options.dataset)
    training_dataset = Data.BigDataset(
        dataset_file, options.training_route, options.samples_name)
    if not options.no_validation:
        validation_dataset = Data.BigDataset(
            dataset_file, options.validation_route, options.samples_name)
    test_dataset = Data.BigDataset(
        dataset_file, options.test_route, options.samples_name)
    n_visible = training_dataset.get_dimensionality(0)

    # # Calculate normalsation constants
    if options.normalize:
        # Normalise all datasets
        mean, std = Data.utils.get_dataset_statistics(training_dataset)
        training_dataset = Data.utils.normalise_dataset(training_dataset, mean, std)
        if not options.no_validation:
            validation_dataset = Data.utils.normalise_dataset(
                validation_dataset, mean, std)
        test_dataset = Data.utils.normalise_dataset(test_dataset, mean, std)
        hdf5_backend.write([], "normalisation/mean", mean)
        hdf5_backend.write([], "normalisation/std", std)
    # Dataset of masks
    try:
        masks_filename = options.dataset + "." + floatX + ".masks"
        masks_route = os.path.join(os.environ["DATASETSPATH"], masks_filename)
        masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")
    except:
        create_dropout_masks(os.environ["DATASETSPATH"],
                             masks_filename, n_visible, ks=1000)
        masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")
    return training_dataset, validation_dataset, test_dataset, masks_dataset


def set_loss(options, l, n_visible, validation_dataset, masks_dataset):
    """
    Set loss measurement depending on the 'form' passed in.
    """
    if options.form == "MoG":
        nade_class = NADE.OrderlessMoGNADE
        nade = nade_class(n_visible, options.units, l,
                          options.n_components, nonlinearity=options.nonlinearity)
        loss_function = "sym_masked_neg_loglikelihood_gradient"
        validation_loss_measurement = Instrumentation.Function(
            "validation_loss", lambda ins: -ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops=options.validation_loops))
    elif options.form == "Bernoulli":
        nade_class = NADE.OrderlessBernoulliNADE
        nade = nade_class(n_visible, options.units, l, nonlinearity=options.nonlinearity)
        loss_function = "sym_masked_neg_loglikelihood_gradient"
        validation_loss_measurement = Instrumentation.Function(
            "validation_loss", lambda ins: -ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops=options.validation_loops))
    elif options.form == "QR":
        nade_class = NADE.OrderlessQRNADE
        nade = nade_class(n_visible, options.units, l,
                          options.n_quantiles, nonlinearity=options.nonlinearity)
        loss_function = "sym_masked_pinball_loss_gradient"
        validation_loss_measurement = Instrumentation.Function("validation_loss", lambda ins: ins.model.estimate_average_pinball_loss_for_dataset(
            validation_dataset, masks_dataset, loops=options.validation_loops))
    else:
        raise Exception("Unknown form")

    return nade_class, nade, loss_function, validation_loss_measurement


def pretrain_layerwise(m):
    """
    """
    options = m.options
    textfile_log = m.textfile_log
    hdf5_backend = m.hdf5_backend
    console = m.console
    nade_class = m.nade_class
    nade = m.nade
    loss_function = m.loss_function
    training_dataset = m.training_dataset
    masks_dataset = m.masks_dataset

    # Pretrain layerwise
    for l in xrange(1, options.hlayers + 1):
        if l == 1:
            nade.initialize_parameters_from_dataset(training_dataset)
        else:
            nade = nade_class.create_from_smaller_NADE(nade, add_n_hiddens=1)
        # Configure training
        trainer = Optimization.MomentumSGD(
            nade, nade.__getattribute__(loss_function))
        trainer.set_datasets([training_dataset, masks_dataset])
        trainer.set_learning_rate(options.lr)
        trainer.set_datapoints_as_columns(True)
        trainer.add_controller(TrainingController.AdaptiveLearningRate(
            options.lr, 0, epochs=options.pretraining_epochs))
        trainer.add_controller(
            TrainingController.MaxIterations(options.pretraining_epochs))
        trainer.add_controller(TrainingController.ConfigurationSchedule(
            "momentum", [(2, 0), (float('inf'), options.momentum)]))
        trainer.set_updates_per_epoch(options.epoch_size)
        trainer.set_minibatch_size(options.batch_size)
    #    trainer.set_weight_decay_rate(options.wd)
        trainer.add_controller(TrainingController.NaNBreaker())
        # Instrument the training
        trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend],
                                                                    Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
        trainer.add_instrumentation(Instrumentation.Instrumentation(
            [console, textfile_log, hdf5_backend], Instrumentation.Configuration()))
        trainer.add_instrumentation(Instrumentation.Instrumentation(
            [console, textfile_log, hdf5_backend], Instrumentation.Timestamp()))
        # Train
        trainer.set_context("pretraining_%d" % l)
        trainer.train()
    return nade, trainer


class ModelMngr():

    def __init__(self, options):
        print("Init model manager")
        self.options = options


def pre_training_configs(m):
    """
    Before training the model, configure it 
    """
    ordering = range(m.n_visible)
    np.random.shuffle(ordering)
    trainer = Optimization.MomentumSGD(m.nade, m.nade.__getattribute__(m.loss_function))
    trainer.set_datasets([m.training_dataset, m.masks_dataset])
    trainer.set_learning_rate(m.options.lr)
    trainer.set_datapoints_as_columns(True)
    trainer.add_controller(TrainingController.AdaptiveLearningRate(
        m.options.lr, 0, epochs=m.options.epochs))
    trainer.add_controller(TrainingController.MaxIterations(m.options.epochs))
    if m.options.training_ll_stop < np.inf:
        # Assumes that we're doing minimization so negative ll
        trainer.add_controller(
            TrainingController.TrainingErrorStop(-m.options.training_ll_stop))
    trainer.add_controller(TrainingController.ConfigurationSchedule(
        "momentum", [(2, 0), (float('inf'), m.options.momentum)]))
    trainer.set_updates_per_epoch(m.options.epoch_size)
    trainer.set_minibatch_size(m.options.batch_size)
#    trainer.set_weight_decay_rate(options.wd)
    trainer.add_controller(TrainingController.NaNBreaker())
    # Instrument the training
    trainer.add_instrumentation(Instrumentation.Instrumentation(
        [m.console, m.textfile_log, m.hdf5_backend], Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
    if not m.options.no_validation:
        trainer.add_instrumentation(Instrumentation.Instrumentation([m.console],
                                                                    m.validation_loss_measurement))
        trainer.add_instrumentation(Instrumentation.Instrumentation([m.hdf5_backend],
                                                                    m.validation_loss_measurement,
                                                                    at_lowest=[Instrumentation.Parameters()]))
    trainer.add_instrumentation(Instrumentation.Instrumentation(
        [m.console, m.textfile_log, m.hdf5_backend], Instrumentation.Configuration()))
    # trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend], Instrumentation.Parameters(), every = 10))
    trainer.add_instrumentation(Instrumentation.Instrumentation(
        [m.console, m.textfile_log, m.hdf5_backend], Instrumentation.Timestamp()))

    return trainer


def main():
    parser = get_parser()
    (options, args) = parser.parse_args()
    # if options.theano:
    #     import NADE
    # else:
    #     import npNADE as NADE
    #     raise Exception("Not implemented yet")

    results_route = os.path.join(os.environ["RESULTSPATH"], args[0])
    try:
        os.makedirs(results_route)
    except OSError:
        pass

    console = Backends.Console()
    textfile_log = Backends.TextFile(os.path.join(results_route, "NADE_training.log"))
    hdf5_backend = Backends.HDF5(results_route, "NADE")
    hdf5_backend.write([], "options", options)
    hdf5_backend.write([], "svn_revision", Utils.svn.svnversion())
    hdf5_backend.write([], "svn_status", Utils.svn.svnstatus())
    hdf5_backend.write([], "svn_diff", Utils.svn.svndiff())

    training_dataset, validation_dataset, test_dataset, masks_dataset = prep_datasets(
        options, hdf5_backend)

    n_visible = training_dataset.get_dimensionality(0)

    l = 1 if options.layerwise else options.hlayers

    nade_class, nade, loss_function, validation_loss_measurement = set_loss(
        options, l, n_visible, validation_dataset, masks_dataset)

    # Keep track of all model params in this object, m
    m = ModelMngr(options)
    m.n_visible = n_visible
    m.textfile_log = textfile_log
    m.hdf5_backend = hdf5_backend
    m.console = console
    m.nade_class = nade_class
    m.nade = nade
    m.validation_loss_measurement = validation_loss_measurement
    m.loss_function = loss_function
    m.training_dataset = training_dataset
    m.masks_dataset = masks_dataset

    if m.options.layerwise:
        nade, trainer = pretrain_layerwise(m)
        m.nade = nade
        m.trainer = trainer

    else:  # No pretraining
        nade.initialize_parameters_from_dataset(m.training_dataset)

    # Configure training
    trainer = pre_training_configs(m)

    # Train
    trainer.train()
    #------------------------------------------------------------------------------
    # Report some final performance measurements
    if trainer.was_successful():
        np.random.seed(8341)
        m.hdf5_backend.write(["final_model"], "parameters", nade.get_parameters())
        if not m.options.no_validation:
            nade.set_parameters(m.hdf5_backend.read(
                "/lowest_validation_loss/parameters"))
        config = {"wd": m.options.wd, "h": m.options.units,
                  "lr": m.options.lr, "q": m.options.n_quantiles}
        log_message([console, textfile_log], "Config %s" % str(config))
        if m.options.show_training_stop:
            training_likelihood = nade.estimate_loglikelihood_for_dataset(
                training_dataset)
            log_message([console, textfile_log], "Training average loss\t%f" %
                        training_likelihood)
            m.hdf5_backend.write([], "training_loss", training_likelihood)
        val_ests = []
        test_ests = []
        for i in xrange(m.options.summary_orderings):
            nade.setup_n_orderings(n=1)
            if not m.options.no_validation:
                val_ests.append(
                    nade.estimate_loglikelihood_for_dataset(validation_dataset))
            test_ests.append(nade.estimate_loglikelihood_for_dataset(test_dataset))
        if not m.options.no_validation:
            val_avgs = map(lambda x: x.estimation, val_ests)
            val_mean, val_se = (np.mean(val_avgs), scipy.stats.sem(val_avgs))
            log_message([console, textfile_log],
                        "*Validation mean\t%f \t(se: %f)" % (val_mean, val_se))
            m.hdf5_backend.write([], "validation_likelihood", val_mean)
            m.hdf5_backend.write([], "validation_likelihood_se", val_se)
            for i, est in enumerate(val_ests):
                log_message([console, textfile_log], "Validation detail #%d mean\t%f \t(se: %f)" % (
                    i + 1, est.estimation, est.se))
                m.hdf5_backend.write(["results", "orderings", str(i + 1)],
                                     "validation_likelihood", est.estimation)
                m.hdf5_backend.write(["results", "orderings", str(i + 1)],
                                     "validation_likelihood_se", est.se)
        test_avgs = map(lambda x: x.estimation, test_ests)
        test_mean, test_se = (np.mean(test_avgs), scipy.stats.sem(test_avgs))
        log_message([console, textfile_log], "*Test mean\t%f \t(se: %f)" %
                    (test_mean, test_se))
        m.hdf5_backend.write([], "test_likelihood", test_mean)
        m.hdf5_backend.write([], "test_likelihood_se", test_se)
        for i, est in enumerate(test_ests):
            log_message([console, textfile_log], "Test detail #%d mean\t%f \t(se: %f)" % (
                i + 1, est.estimation, est.se))
            m.hdf5_backend.write(["results", "orderings", str(i + 1)],
                                 "test_likelihood", est.estimation)
            m.hdf5_backend.write(["results", "orderings", str(i + 1)],
                                 "test_likelihood_se", est.se)
        m.hdf5_backend.write([], "final_score", test_mean)
        # Report results on ensembles of NADES
        if m.options.report_mixtures:
            # #
            for components in [2, 4, 8, 16, 32]:
                nade.setup_n_orderings(n=components)
                est = nade.estimate_loglikelihood_for_dataset(test_dataset)
                log_message([console, textfile_log], "Test ll mixture of nades %d components: mean\t%f \t(se: %f)" % (
                    components, est.estimation, est.se))
                m.hdf5_backend.write(["results", "mixtures", str(
                    components)], "test_likelihood", est.estimation)
                m.hdf5_backend.write(["results", "mixtures", str(
                    components)], "test_likelihood_se", est.se)


if __name__ == "__main__":
    main()
