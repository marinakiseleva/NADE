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

        print(trainable.epoch)
        if trainable.epoch < 5:
            print("don't count this")
            return False
        vloss = -self.nade.estimate_loglikelihood_for_dataset(
            self.validation_dataset, minibatch_size=20)

        training_loss = trainable.get_training_loss()

        print("\nValidation loss " + str(vloss) +
              "\nTraining loss " + str(training_loss) + "\n\n")

        if training_loss < vloss:
            return True
