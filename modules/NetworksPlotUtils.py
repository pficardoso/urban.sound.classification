
import matplotlib.pyplot as plt
import numpy as np


def plot_models_loss_acc_history(models_history, min_loss=0, max_loss=5, step_loss=1, min_acc=0, max_acc=1, step_acc=0.1 ):
    """
    :param models_history: list of "history" object. Each "history" is a dictionary with lists as values
    :param min_loss: scalar
    :param max_loss: scalar
    :param step_loss: scalar
    :param min_eval_score: scalar
    :param max_eval_score: scalar
    :param step_eval_score: scalar
    :return:
    """

    if isinstance(models_history, list):

        fig, ax = plt.subplots( nrows=len(models_history),
                                ncols=2,
                                figsize=(10 , 4 * len(models_history)))

        for i, history in enumerate(models_history):
            train_loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            train_acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            epochs = range(1, len(train_loss ) +1)

            ## plot loss
            ax[i][0].plot(epochs, train_loss)
            ax[i][0].plot(epochs, val_loss)
            ax[i][0].legend(("train", "validation"))
            ax[i][0].set_title("Loss")
            ax[i][0].yaxis.set_ticks( np.arange(min_loss , max_loss +0.1, step_loss ) )
            ax[i][0].set_ylim(min_loss, max_loss)

            ## plot accuracy
            ax[i][1].plot(epochs, train_acc)
            ax[i][1].plot(epochs, val_acc)
            ax[i][1].legend(("train", "validation"))
            ax[i][1].set_title("Accuracy")
            ax[i][1].yaxis.set_ticks( np.arange( min_acc, max_acc +0.1 , step_acc) )
            ax[i][1].set_ylim(min_acc, max_acc)

            ##get minimizer of loss
            minimizer_train_loss, minimizer_val_loss   = [np.argmin([loss]) for loss in [train_loss, val_loss]]

            ##get maximizer of acc
            maximizer_train_acc, maximizer_val_acc = [np.argmax([acc]) for acc in [train_acc, val_acc]]

            print("Model", i+1)
            print("At minimizer of train's loss (i = {}): loss = {:.2f}; acc = {:.2f}".format(minimizer_train_loss,
                                                                                     train_loss[minimizer_train_loss],
                                                                                     train_acc[minimizer_train_loss]))

            print("At minimizer of validations's loss (i = {}): loss = {:.2f}; acc = {:.2f}".format(minimizer_val_loss,
                                                                                            val_loss[minimizer_val_loss],
                                                                                            val_acc[minimizer_train_loss]))

            print("At maximizer of train's accuracy (i = {}): loss = {:.2f}; acc = {:.2f}".format(maximizer_train_acc,
                                                                                          train_loss[maximizer_train_acc],
                                                                                          train_acc[maximizer_train_acc]))

            print("At maximizer of validations's accuracy (i = {}):  loss = {:.2f}; acc = {:.2f}".format(maximizer_val_acc,
                                                                                                 val_loss[maximizer_val_acc],
                                                                                                 val_acc[maximizer_val_acc]))
            print("\n")

    else:
        fig, ax = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(10, 4))
        history = models_history
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        train_acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        epochs = range(1, len(train_loss) + 1)

        ## plot loss
        ax[0].plot(epochs, train_loss)
        ax[0].plot(epochs, val_loss)
        ax[0].legend(("train", "validation"))
        ax[0].set_title("Loss")
        ax[0].yaxis.set_ticks(np.arange(min_loss, max_loss + 0.1, step_loss))
        ax[0].set_ylim(min_loss, max_loss)

        ## plot accuracy
        ax[1].plot(epochs, train_acc)
        ax[1].plot(epochs, val_acc)
        ax[1].legend(("train", "validation"))
        ax[1].set_title("Accuracy")
        ax[1].yaxis.set_ticks(np.arange(min_acc, max_acc + 0.1, step_acc))
        ax[1].set_ylim(min_acc, max_acc)

        ##get minimizer of loss
        minimizer_train_loss, minimizer_val_loss = [np.argmin([loss]) for loss in [train_loss, val_loss]]

        ##get maximizer of acc
        maximizer_train_acc, maximizer_val_acc = [np.argmax([acc]) for acc in [train_acc, val_acc]]

        print("At minimizer of train's loss (i = {}): loss = {:.2f}; acc = {:.2f}".format(minimizer_train_loss,
                                                                                          train_loss[
                                                                                              minimizer_train_loss],
                                                                                          train_acc[
                                                                                              minimizer_train_loss]))

        print("At minimizer of validations's loss (i = {}): loss = {:.2f}; acc = {:.2f}".format(minimizer_val_loss,
                                                                                                val_loss[
                                                                                                    minimizer_val_loss],
                                                                                                val_acc[
                                                                                                    minimizer_train_loss]))

        print("At maximizer of train's accuracy (i = {}): loss = {:.2f}; acc = {:.2f}".format(maximizer_train_acc,
                                                                                              train_loss[
                                                                                                  maximizer_train_acc],
                                                                                              train_acc[
                                                                                                  maximizer_train_acc]))

        print("At maximizer of validations's accuracy (i = {}):  loss = {:.2f}; acc = {:.2f}".format(
            maximizer_val_acc,
            val_loss[maximizer_val_acc],
            val_acc[maximizer_val_acc]))