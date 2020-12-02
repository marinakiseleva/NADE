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

        vloss = self.nade.estimate_loglikelihood_for_dataset(
            self.validation_dataset, minibatch_size=20)
        print("\n")
        print("Validation loss " + str(vloss))
        training_loss = -trainable.get_training_loss()
        print("Training loss " + str(training_loss))
        print("\n\n")

        # if trainable.get_training_loss() < v:
        #     print("\n\n training is better than validation \n\n")
