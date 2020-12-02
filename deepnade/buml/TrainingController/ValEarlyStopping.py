from .TrainingController import TrainingController


class ValEarlyStopping(TrainingController):
    """
    Stops the training when the training error is lower than the validation error 

    """

    def __init__(self, nade, validation_dataset):
        """
        :param v: Negative log likelihood
        """
        print("Using Early Stopping on Validation Data.")
        self.nade = nade
        self.validation_dataset = validation_dataset

    def after_training_iteration(self, trainable):

        if trainable.epoch < 5:
            print("don't count this")
            return False
        #  log likelihood validation set

        nll_validation = -self.nade.estimate_loglikelihood_for_dataset(
            self.validation_dataset, minibatch_size=20)

        # negative log likelihood of training set (pretty sure)
        nll_training = trainable.get_training_loss()

        print("\nValidation loss (negative)" + str(nll_validation) +
              "\nTraining loss (negative)" + str(nll_training) + "\n\n")

        if nll_training < nll_validation:
            return True
