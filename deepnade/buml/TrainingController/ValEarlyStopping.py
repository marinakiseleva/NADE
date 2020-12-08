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
        self.last_X = []
        self.X = 10

    def after_training_iteration(self, trainable):

        if trainable.epoch < 10:
            print("Don't count this")
            return False

        ll_validation = self.nade.estimate_loglikelihood_for_dataset(
            self.validation_dataset, minibatch_size=20)

        ll_training = self.nade.estimate_loglikelihood_for_dataset(
            self.training_dataset, minibatch_size=20)

        print("\nValidation ll" + str(ll_validation) +
              "\nTraining ll" + str(ll_training) + "\n\n")

        # Note when validation likelihood is better than training
        if ll_training < ll_validation:
            self.last_X.append(True)

        # When it happens last_X times, stop.
        if len(self.last_X) >= self.X:
            return True
