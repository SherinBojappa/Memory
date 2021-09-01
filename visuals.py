import pickle
import matplotlib.pyplot as plt
import numpy as np
max_seq_len = 100
# load the pickle file

# read the pickle file cnn
f_cnn = open('balanced_acc_seq_len_CNN.pkl', 'rb')
acc_cnn = pickle.load(f_cnn)
f_cnn.close()

# read the pickle file rnn
f_rnn = open('balanced_acc_seq_len_RNN.pkl', 'rb')
acc_rnn = pickle.load(f_rnn)
f_rnn.close()

# read the pickle file rnn
f_lstm = open('balanced_acc_seq_len_lstm.pkl', 'rb')
acc_lstm = pickle.load(f_lstm)
f_lstm.close()

"""
# plot the seq len vs accuracy
x = np.arange(1,max_seq_len)
plt.title("Balanced accuracy versus sequence length")
plt.xlabel("sequence length")
plt.ylabel("balanced accuracy")
plot_rnn, = plt.plot(x, acc_rnn[1:100], color='green', label='RNN')
plot_cnn, = plt.plot(x, acc_cnn[1:100], color='blue', label='CNN')
plot_lstm, = plt.plot(x, acc_lstm[1:100], color='magenta', label='LSTM')
plt.legend(handles=[plot_rnn, plot_cnn, plot_lstm])
plt.savefig("Models_acc_vs_seq_len")
"""

f_lstm = open('balanced_acc_seq_len_dist_lstm.pkl', 'rb')
acc_lstm = pickle.load(f_lstm)
f_lstm.close()

# plot the balanced accuracy per sequence length
# ax1 is for 3d and imshow is for 2d
fig=plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
x = np.arange(0,max_seq_len-1)
y = np.arange(0, max_seq_len)
xs, ys = np.meshgrid(y, x)
plt.title("Balanced accuracy versus sequence length")
plt.ylabel("sequence length")
plt.xlabel("distance")
surf = ax1.plot_surface(xs, ys, np.array(acc_lstm), cmap='plasma')
fig.colorbar(surf, ax=ax1)

#plt.imshow(acc_lstm)
#plt.colorbar()
plt.show()



