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
from tensorflow.keras.utils import to_categorical

# dataset is fra.txt which is downloaded from http://www.manythings.org/anki/fra-eng.zip

batch_size = 64  # Batch size for training.
#batch_size = 5
#epochs = 5  # Number of epochs to train for.
epochs = 1
latent_dim = 256  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
data_path = "fra.txt"
input_seq = 'synthetic'
seq_len = 4
num_repeat = 1
num_tokens_rep = 1
max_seq_len = 600
eos_encoder = np.zeros(max_seq_len+1)
eos_encoder[0] = 1
eos_decoder = 2
sos_decoder = 3
verbose = 0

x, y, y_mlp, token_repeated, pos_first_token, sequence_len = generate_dataset(max_seq_len,
                                                                      num_tokens_rep)

# plots on the properties of the generated sequences
#plot_data(x, y, token_repeated, pos_first_token, repeat_dist, repeat_position)

num_samples = len(x)
print("The total number of samples in the dataset is " + str(num_samples))

# decoder tokens = 0,1,eos
num_decoder_tokens = 3
# encoder tokens = 26 tokens in alphabet + eos
num_encoder_tokens = max_seq_len+1

# the data x,y already have eos appended to them
max_encoder_seq_length = max([len(seq) for seq in x])
max_decoder_seq_length = max([len(seq) for seq in y])
print("Max encoder length" + str(max_encoder_seq_length))
print("Max decoder length" + str(max_decoder_seq_length))

# seq len + 1 for alphabet + eos
encoder_input_data = np.zeros((num_samples, max_encoder_seq_length,
                               num_encoder_tokens), dtype="float32")

if(verbose == 1):
    decode_seq(x, y)

one_hot_encoding_label = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]])

for i in range(num_samples):
    seq_len = len(x[i])
    for seq in range(seq_len):
        # sos for decoder_input_data
        encoder_input_data[i, seq] = x[i][seq]

# train test split
(encoder_input_data_train, encoder_input_data_test,
y_mlp_train, y_mlp_test,
sequence_len_train, sequence_len_test,
token_repeated_train, token_repeated_test,
pos_first_token_train, pos_first_token_test) = train_test_split(encoder_input_data, y_mlp,
                                 sequence_len, token_repeated, pos_first_token, test_size=0.3)

# num_train = 0.8*encoder_input_data.shape[0]
# encoder_input_data_train = encoder_input_data[0:int(num_train)][:][:]
# decoder_input_data_train = decoder_input_data[0:int(num_train)][:][:]
# decoder_target_data_train = decoder_target_data[0:int(num_train)][:][:]
# sequence_len_train = sequence_len[0:int(num_train)]
#
# num_test = 0.2*encoder_input_data.shape[0]
# encoder_input_data_test = encoder_input_data[int(num_train):][:][:]
# decoder_input_data_test = decoder_input_data[int(num_train):][:][:]
# decoder_target_data_test = decoder_target_data[int(num_train):][:][:]
# sequence_len_test = sequence_len[int(num_train):]


# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = tf.concat((state_h, state_c),1)
num_classes=2
input_shape = encoder_states.shape[1]

model_mlp = Sequential()
model_mlp.add(Dense(350, input_shape=(None, input_shape), activation='relu'))
model_mlp.add(Dense(50, activation='relu'))
model_mlp.add(Dense(num_classes, activation='softmax'))

mlp_output = model_mlp(encoder_states)
# divide into training and test sets 80% and 20%

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model(encoder_inputs, mlp_output)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# early stopping
es_cb = EarlyStopping(monitor="val_loss", patience=0, verbose=1,
                      mode="min")

y_mlp_binary_train = to_categorical(np.array(y_mlp_train), dtype="float32")

history = model.fit(
    encoder_input_data_train,
    y_mlp_binary_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.3,
    callbacks=[es_cb]
)

print("Number of epochs run: " + str(len(history.history["loss"])))


y_true = np.array(y_mlp_test, dtype="float32")
# test results
y_test = model.predict(encoder_input_data_test)
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




