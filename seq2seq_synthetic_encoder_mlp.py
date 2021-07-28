# https://keras.io/examples/nlp/lstm_seq2seq/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
#from dataset.memory_dataset_generation import *
from dataset.synthetic_dataset_encoder_mlp import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D

# dataset is fra.txt which is downloaded from http://www.manythings.org/anki/fra-eng.zip

batch_size = 64  # Batch size for training.
#batch_size = 5
#epochs = 5  # Number of epochs to train for.
epochs = 10
latent_dim = 256  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
data_path = "fra.txt"
input_seq = 'synthetic'
seq_len = 4
num_repeat = 1
num_tokens_rep = 1
max_seq_len = 26
eos_encoder = np.zeros(max_seq_len+1)
eos_encoder[0] = 1
eos_decoder = 2
sos_decoder = 3
verbose = 0

# memory model can be lstm, rnn, or cnn
memory_model = "lstm"
#memory_model = "CNN"
#memory_model = "RNN"

x, y, y_mlp, token_repeated, pos_first_token, sequence_len = generate_dataset(max_seq_len,
                                                                      num_tokens_rep)

num_samples = len(x)
print("The total number of samples in the dataset is " + str(num_samples))

# check if the train and test have different inputs
if(verbose == 1):
    num_rept_test = check_train_test(x)
    print("The number of overlaps in train and test are : " + str(num_rept_test))

# separate out the input to the encoder and the mlp
# mlp is fed the last one hot encoded input
x_mlp = [0]*num_samples
x_encoder = [0]*num_samples

for iter, seq in enumerate(x):
    x_mlp[iter] = seq[-2]
    # all but the last one hot encoded sequence
    x_encoder[iter] = seq[0:-2]
    #eos
    x_encoder[iter].append(seq[-1])

# plots on the properties of the generated sequences
#plot_data(x, y, token_repeated, pos_first_token, repeat_dist, repeat_position)



# decoder tokens = 0,1,eos
num_decoder_tokens = 3
# encoder tokens = 26 tokens in alphabet + eos
num_encoder_tokens = max_seq_len+1

# the data x,y already have eos appended to them
max_encoder_seq_length = max([len(seq) for seq in x_encoder])
max_decoder_seq_length = max([len(seq) for seq in y])
print("Max encoder length" + str(max_encoder_seq_length))
print("Max decoder length" + str(max_decoder_seq_length))

# seq len + 1 for alphabet + eos
encoder_input_data = np.zeros((num_samples, max_encoder_seq_length,
                               num_encoder_tokens), dtype="float32")

mlp_input_data = np.zeros((num_samples, num_encoder_tokens), dtype = "float32")

one_hot_encoding_label = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]])

for i in range(num_samples):
    seq_len = len(x_encoder[i])
    for seq in range(seq_len):
        encoder_input_data[i, seq] = x_encoder[i][seq]
    mlp_input_data[i] = x_mlp[i]

# train test split
(encoder_input_data_train, encoder_input_data_test,
 mlp_input_data_train, mlp_input_data_test,
y_mlp_train, y_mlp_test,
sequence_len_train, sequence_len_test,
token_repeated_train, token_repeated_test,
pos_first_token_train, pos_first_token_test) = train_test_split(encoder_input_data, mlp_input_data, y_mlp,
                                 sequence_len, token_repeated, pos_first_token, test_size=0.3)


if(verbose == 1):
    check_dataset(encoder_input_data_train, encoder_input_data_test,
                  mlp_input_data_train, mlp_input_data_test,
                  y_mlp_train, y_mlp_test,
                  sequence_len_train, sequence_len_test,
                  token_repeated_train, token_repeated_test,
                  pos_first_token_train, pos_first_token_test)

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
mlp_input = keras.Input(shape=(num_encoder_tokens))

if(memory_model == "lstm"):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    mlp_input = keras.Input(shape=(num_encoder_tokens))

    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = tf.concat((state_h, state_c), 1)
    print("Encoder chosen is LSTM")
elif(memory_model == "RNN"):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    mlp_input = keras.Input(shape=(num_encoder_tokens))

    encoder = keras.layers.SimpleRNN(latent_dim, return_state=True)
    encoder_output, state = encoder(encoder_inputs)
    encoder_states = encoder_output
    print("Encoder chosen is simple RNN")
    print("Shape of the encoder output is: " + str(encoder_states))
# FIXME BROKEN CNN RESHAPE
elif(memory_model == "CNN"):
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    mlp_input = keras.Input(shape=(num_encoder_tokens))
    encoder = Sequential()
    #encoder.add(keras.layers.Reshape((1, num_encoder_tokens * (num_encoder_tokens - 1))))
    encoder.add(keras.layers.Reshape((1, num_encoder_tokens*(num_encoder_tokens+1))))
    encoder.add(keras.layers.Conv1D(filters = latent_dim, kernel_size = 1, activation='relu'))
    #encoder.add(MaxPooling1D(pool_size=2))

    # flatten makes the shape as [None, None]
    #encoder.add(Flatten())
    encoder.add(keras.layers.Reshape((latent_dim,)))
    # before feeding the input reshape it to be 27
    encoder_output = encoder(encoder_inputs)
    encoder_states = encoder_output
    print("Encoder chosen is CNN")
    print("Shape of the encoder output is: " + str(encoder_states))

mlp_encoder = Sequential()
mlp_ip_shape = mlp_input_data_train.shape[1]
mlp_encoder.add(Dense(latent_dim, input_shape=(mlp_ip_shape,)))
mlp_encoded_op = mlp_encoder(mlp_input)


num_classes=2
input_shape = encoder_states.shape[1]

soft_max_input = tf.concat((encoder_states, mlp_encoded_op), 1)
soft_max_shape = soft_max_input.shape[1]

model_mlp = Sequential()
#model_mlp.add(Dense(num_classes, input_shape = (None, soft_max_shape), activation='softmax'))
model_mlp.add(Dense(50, input_shape=(None, soft_max_shape), activation='relu'))
model_mlp.add(Dense(num_classes, activation='softmax'))

mlp_output = model_mlp(soft_max_input)
# divide into training and test sets 80% and 20%

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, mlp_input], mlp_output)
model.summary()
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# early stopping
es_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=1,
                      mode="min")

y_mlp_binary_train = to_categorical(np.array(y_mlp_train), dtype="float32")

history = model.fit(
    [encoder_input_data_train, mlp_input_data_train],
    y_mlp_binary_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.3,
    callbacks=[es_cb]
)

print("Number of epochs run: " + str(len(history.history["loss"])))


y_true = np.array(y_mlp_test, dtype="float32")
# test results
y_test = model.predict([encoder_input_data_test, mlp_input_data_test])
y_pred = np.argmax(y_test, axis=1)

# total balanced accuracy accross the entire test dataset
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(balanced_accuracy)

# Find the balanced accuracy accross different sequence length
sequence_len_arr = np.array(sequence_len_test)
# balanced_acc_seq_len of 0 and 1 are meaningless
balanced_acc_seq_len = [0]*(max_seq_len+1)

for seq_len in range(2,max_seq_len+1):
    seq_len_indices = []
    # get the indices of samples which have a particular sequence length
    seq_len_indices = np.where(sequence_len_arr == seq_len)
    # splice y_true and y_pred based on the seq length
    y_true_seq_len = np.take(y_true, seq_len_indices)
    y_pred_seq_len = np.take(y_pred, seq_len_indices)
    balanced_acc_seq_len[seq_len] = balanced_accuracy_score(y_true_seq_len[0],
                                                            y_pred_seq_len[0])
    print("Balanced accuracy for seq len {} is {}".format(seq_len, balanced_acc_seq_len[seq_len]))

# plot the balanced accuracy per sequence length
x = np.arange(0,max_seq_len+1)
plt.title("Balanced accuracy versus sequence length")
plt.xlabel("sequence length")
plt.ylabel("balanced accuracy")
plt.plot(x, balanced_acc_seq_len)
plt.show()




