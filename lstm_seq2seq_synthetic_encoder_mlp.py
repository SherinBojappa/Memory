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
max_seq_len = 26
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

history = model.fit(
    encoder_input_data_train,
    y_mlp_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.3,
    callbacks=[es_cb]
)

print("Number of epochs run: " + str(len(history.history["loss"])))

#reverse_target_char_index = [0,1,2,3]
def decode_sequence(input_seq, len_input_sequence):
    # Encode the input as state vectors.
    states_value = encoder_mlp_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 3))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, :] = one_hot_encoding_label[sos_decoder]
    #target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_char = str(reverse_target_char_index[sampled_token_index])
        decoded_sentence.append(sampled_token_index) #sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_token_index == eos_decoder or len(
                decoded_sentence) > len_input_sequence:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 3))
        target_seq[0, 0, :] = one_hot_encoding_label[sampled_token_index]

        # Update states
        states_value = [h, c]
    return np.array(decoded_sentence)

#y_pred = np.zeros((len(encoder_input_data_test), max_decoder_seq_length+1, 3), dtype="float32")
balanced_accuracy = np.zeros((len(encoder_input_data_test),3))

one_hot_encoding_label = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
for seq_index in range(len(encoder_input_data_test)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data_test[seq_index: seq_index + 1]
    len_input_sequence = np.array(sequence_len_test[seq_index: seq_index + 1])
    decoded_sentence = decode_sequence(input_seq, len_input_sequence)
    #val_one_hot = np.zeros((1,max_decoder_seq_length+1,3),dtype="float32")
    #for num, val in enumerate(decoded_sentence):
    #    val_one_hot[0][num][:] = one_hot_encoding_label[int(val)]
    y_true = decoder_target_data_test[seq_index][:len_input_sequence[0]+1].argmax(axis=1).ravel()
    y_est = np.zeros_like(y_true) + 2
    y_est[:len(decoded_sentence)] = decoded_sentence #val_one_hot.argmax(axis=2).ravel()
    metric = recall_score(y_true, y_est, average=None)
    if(len(metric) == 3):
        balanced_accuracy[seq_index] = metric
    else:
        balanced_accuracy[seq_index][0] = metric[0]
        balanced_accuracy[seq_index][1] = 0
        balanced_accuracy[seq_index][2] = metric[1]

    #balanced_accuracy[seq_index] = recall_score(y_true, y_est, average=None)
    print(balanced_accuracy[seq_index])

    #y_pred[seq_index:seq_index+1] = val_one_hot
    """
    print("-")
    print("Decoded sentence:", decoded_sentence)
    seq = []
    for num in range(seq_len):
        seq.append(one_hot_decoding(alphabet, input_seq[0][num][:]))
    print("Input sentence:", seq)
    """
print("Balanced accuracy of test set")
print(np.average(balanced_accuracy,axis=0))
#y_true = decoder_target_data_test.argmax(axis=2).ravel()
#y_est = y_pred.argmax(axis=2).ravel()
#print(balanced_accuracy_score(y_true, y_est))
#print(classification_report(y_true, y_est))




