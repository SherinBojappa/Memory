import pickle
import matplotlib.pyplot as plt
import numpy as np
max_seq_len = 100
plot_type = '2d'
#plot_type = '3d'
#plot_type = '2d_image'
# load the pickle file
"""
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

# @@@@@@@@@@LSTM
"""
f_lstm = open('balanced_acc_seq_len_dist_lstm.pkl', 'rb')
acc_lstm = pickle.load(f_lstm)
f_lstm.close()
"""

f_lstm = open('acc_lstm.pkl', 'rb')
acc_model_lstm = pickle.load(f_lstm)
f_lstm.close()

f_cnn = open('acc_cnn.pkl', 'rb')
acc_model_cnn = pickle.load(f_cnn)
f_cnn.close()

f_rnn = open('acc_rnn.pkl', 'rb')
acc_model_rnn = pickle.load(f_rnn)
f_rnn.close()

f_trans = open('acc_trans.pkl', 'rb')
acc_model_trans = pickle.load(f_trans)
f_trans.close()
"""
f_trans = open('balanced_acc_seq_len_dist_transformer_no_orthonormal.pkl', 'rb')
acc_trans = pickle.load(f_trans)
f_trans.close()
"""

"""
for y in range(0, max_seq_len):
    for x in range(0, max_seq_len-1):
        print(acc_lstm[x][y])

exit()
"""
""" TODO UNCOMMENT ME
# plot the balanced accuracy per sequence length
fig=plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
x = np.arange(0,max_seq_len-1)
y = np.arange(0, max_seq_len)
xs, ys = np.meshgrid(y, x)
plt.title("Balanced accuracy versus sequence length")
plt.ylabel("sequence length")
plt.xlabel("distance")
#plt.zlabel("Distance ")
surf = ax1.plot_surface(xs, ys, np.array(acc_lstm), cmap='plasma')
fig.colorbar(surf, ax=ax1)

#plt.imshow(acc_lstm)
#plt.colorbar()
plt.show()
"""

# plot the sequence length vs accuracy
acc_seq_len_lstm = np.zeros(shape=(max_seq_len)) #[0]*(max_seq_len+1)
acc_seq_len_cnn = np.zeros(shape=(max_seq_len)) #[0]*(max_seq_len+1)
acc_seq_len_rnn = np.zeros(shape=(max_seq_len)) #[0]*(max_seq_len+1)
acc_seq_len_trans = np.zeros(shape=(max_seq_len)) #[0]*(max_seq_len+1)

acc_dist_lstm = np.zeros(shape = (max_seq_len))
acc_dist_cnn = np.zeros(shape = (max_seq_len))
acc_dist_rnn = np.zeros(shape = (max_seq_len))
acc_dist_trans = np.zeros(shape = (max_seq_len))

acc_dist_lstm = np.zeros(shape = (max_seq_len))
acc_dist_cnn = np.zeros(shape = (max_seq_len))
acc_dist_rnn = np.zeros(shape = (max_seq_len))
acc_dist_trans = np.zeros(shape = (max_seq_len))
#for seq_len_idx, acc in enumerate(acc_trans):
for seq_len_idx in range(1, max_seq_len):
    accuracy_lstm = 0
    accuracy_rnn = 0
    accuracy_cnn = 0
    accuracy_trans = 0
    acc_lstm = acc_model_lstm[seq_len_idx]
    acc_rnn = acc_model_rnn[seq_len_idx]
    acc_cnn = acc_model_cnn[seq_len_idx]
    acc_trans = acc_model_trans[seq_len_idx]
    #for dist_idx, dist in enumerate(acc):
    for dist_idx in range(0, seq_len_idx):
        accuracy_lstm += acc_lstm[dist_idx]
        accuracy_cnn += acc_cnn[dist_idx]
        accuracy_rnn += acc_rnn[dist_idx]
        accuracy_trans += acc_trans[dist_idx]
    acc_seq_len_lstm[seq_len_idx] = accuracy_lstm/(dist_idx+1)
    acc_seq_len_rnn[seq_len_idx] = accuracy_rnn/(dist_idx+1)
    acc_seq_len_cnn[seq_len_idx] = accuracy_cnn/(dist_idx+1)
    acc_seq_len_trans[seq_len_idx] = accuracy_trans/(dist_idx+1)
    #print("For given sequence length {} the accuracy is {}".format(seq_len_idx, acc_seq_len[seq_len_idx]))

# plot the intermediate tokens - dist- vs accuracy
for dist_idx in range(0, max_seq_len):
    accuracy_lstm = 0
    accuracy_rnn = 0
    accuracy_cnn = 0
    accuracy_trans = 0
    for seq_len_idx in range(0, max_seq_len):
        accuracy_lstm += acc_model_lstm[seq_len_idx][dist_idx]
        accuracy_rnn += acc_model_rnn[seq_len_idx][dist_idx]
        accuracy_cnn += acc_model_cnn[seq_len_idx][dist_idx]
        accuracy_trans += acc_model_trans[seq_len_idx][dist_idx]
    acc_dist_lstm[dist_idx] = accuracy_lstm/(max_seq_len - dist_idx + 2)
    acc_dist_rnn[dist_idx] = accuracy_rnn/(max_seq_len - dist_idx + 2)
    acc_dist_cnn[dist_idx] = accuracy_cnn/(max_seq_len - dist_idx + 2)
    acc_dist_trans[dist_idx] = accuracy_trans/(max_seq_len - dist_idx + 2)
    print("For a given distance {} the accuracy is {}".format(dist_idx, acc_dist_lstm[dist_idx]))



if plot_type == '2d':
    # plot the seq len vs accuracy
    x = np.arange(1,max_seq_len)
    plt.title("Balanced accuracy versus sequence length")
    plt.xlabel("sequence length")
    plt.ylabel("balanced accuracy")
    plot_rnn, = plt.plot(x, acc_seq_len_rnn[1:100], color='green', label='RNN')
    plot_cnn, = plt.plot(x, acc_seq_len_cnn[1:100], color='blue', label='CNN')
    plot_lstm, = plt.plot(x, acc_seq_len_lstm[1:100], color='magenta', label='LSTM')
    plot_trans, = plt.plot(x, acc_seq_len_trans[1:100], color='cyan', label='TRANSFORMER')
    plt.legend(handles=[plot_rnn, plot_cnn, plot_lstm, plot_trans])

    #plt.savefig("Models_acc_vs_seq_len")
    plt.show()

    # plot the seq len vs accuracy
    x = np.arange(0,max_seq_len-2)
    plt.title("Balanced accuracy versus intervening tokens")
    plt.xlabel("intervening tokens")
    plt.ylabel("balanced accuracy")
    plot_rnn, = plt.plot(x, acc_seq_len_rnn[0:98], color='green', label='RNN')
    plot_cnn, = plt.plot(x, acc_seq_len_cnn[0:98], color='blue', label='CNN')
    plot_lstm, = plt.plot(x, acc_dist_lstm[0:98], color='magenta', label='LSTM')
    plot_trans, = plt.plot(x, acc_dist_trans[0:98], color='cyan', label='TRANSFORMER')
    plt.legend(handles=[plot_rnn, plot_cnn, plot_lstm, plot_trans])

    #plt.savefig("Models_acc_vs_seq_len")
    plt.show()

if plot_type == '3d':
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    x = np.arange(0, max_seq_len)
    y = np.arange(0, max_seq_len)
    xs, ys = np.meshgrid(y, x)
    plt.title("Balanced accuracy versus sequence length and intervening tokens")
    plt.ylabel("sequence length")
    plt.xlabel("intervening tokens")
    # plt.zlabel("Distance ")
    surf = ax1.plot_surface(xs, ys, np.array(acc_model_lstm), cmap='plasma')
    fig.colorbar(surf, ax=ax1)

    # plt.imshow(acc_lstm)
    # plt.colorbar()
    plt.show()

if plot_type == '2d_image':
    fig = plt.figure()
    #ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    x = np.arange(0, max_seq_len)
    y = np.arange(0, max_seq_len)
    xs, ys = np.meshgrid(y, x)
    plt.title("Balanced accuracy versus sequence length and intervening tokens")
    plt.ylabel("sequence length")
    plt.xlabel("intervening tokens")
    # plt.zlabel("Distance ")
    #surf = ax1.plot_surface(xs, ys, np.array(acc_model), cmap='plasma')
    #fig.colorbar(surf, ax=ax1)

    plt.imshow(acc_model_lstm)
    plt.colorbar()
    plt.show()


