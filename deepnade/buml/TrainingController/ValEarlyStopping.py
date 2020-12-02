from .TrainingController import TrainingController


class ValEarlyStopping(TrainingController):
    """
    Stops the training when the training error is lower than the validation error 

    """

    def __init__(self, nade, training_dataset, validation_dataset):
        """
        :param v: Negative log likelihood
        """
        print("Using Early Stopping on Validation Data.")
        self.nade = nade
        self.validation_dataset = validation_dataset
        self.training_dataset = training_dataset

    def after_training_iteration(self, trainable):

        if trainable.epoch < 5:
            print("don't count this")
            return False

        ll_validation = self.nade.estimate_loglikelihood_for_dataset(
            self.validation_dataset, minibatch_size=20)

        ll_training = self.nade.estimate_loglikelihood_for_dataset(
            self.training_dataset, minibatch_size=20)

        print("\nValidation ll" + str(ll_validation) +
              "\nTraining ll" + str(ll_training) + "\n\n")

        # Stop when training likelihood is better than validation
        if ll_training > ll_validation:
            return True
