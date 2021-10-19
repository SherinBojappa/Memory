import numpy as np
import matplotlib.pyplot as plt

def plot_test_val_loss(type):
    # read the test and val loss
    test_loss = np.load('/Users/sherin/Desktop/test_loss_lstm.npy')
    val_loss = np.load('/Users/sherin/Desktop/val_loss_lstm.npy')
    if(type == 'separate'):


        plt.title("Test and validation loss")
        epochs = range(1, 201)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt_test, = plt.plot(epochs, test_loss, color='red', label='Test loss')
        plt_val, = plt.plot(epochs, val_loss, color='cyan', label='Validation loss')
        plt.legend(handles=[plt_test, plt_val])
        plt.show()
    elif (type == 'scatter'):
        plt.title("Test and validation loss")
        plt.xlabel("validation loss")
        plt.ylabel("test loss")
        plt.plot(val_loss, test_loss, 'o', color='magenta');
        plt.show()



def plot_test_val_accuracy(type):
    # read the test and val loss
    test_acc = np.load('/Users/sherin/Desktop/test_accuracy_lstm.npy')
    val_acc = np.load('/Users/sherin/Desktop/val_accuracy_lstm.npy')
    if(type == 'separate'):
        plt.title("Test and validation accuracy")
        epochs = range(1, 201)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt_test, = plt.plot(epochs, test_acc, color='red', label='Test accuracy')
        plt_val, = plt.plot(epochs, val_acc, color='cyan', label='Validation accuracy')
        plt.legend(handles=[plt_test, plt_val])
        plt.show()
    elif (type == 'scatter'):
        plt.title("Test and validation accuracy")
        plt.xlabel("validation accuracy")
        plt.ylabel("test accuracy")
        plt.plot(val_acc, test_acc, 'o', color='magenta');
        plt.show()


#type = 'separate'
type = 'scatter'

plot_test_val_loss(type)
plot_test_val_accuracy(type)



