# https://keras.io/examples/nlp/lstm_seq2seq/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset.memory_dataset_generation import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

# dataset is fra.txt which is downloaded from http://www.manythings.org/anki/fra-eng.zip

batch_size = 64  # Batch size for training.
#batch_size = 5
#epochs = 5  # Number of epochs to train for.
epochs = 5
latent_dim = 256  # Latent dimensionality of the encoding space.
#num_samples = 10000  # Number of samples to train on.
num_samples = 10000
# Path to the data txt file on disk.
data_path = "fra.txt"
#input_seq = 'default'
input_seq = 'synthetic'
seq_len = 4
num_repeat = 1
repeat_dist = 1
num_tokens_rep = 1
# max seq len = 26+eos
max_seq_len = 27
eos_encoder = np.zeros(max_seq_len)
eos_encoder[0] = 1
eos_decoder = 2
sos_decoder = 3
verbose = 0


if input_seq == 'default':
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, _ = line.split("\t")
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0
else:

    num_decoder_tokens = 3
    num_encoder_tokens = max_seq_len
    # seq len + 1 for alphabet + eos
    encoder_input_data = np.zeros((num_samples, seq_len+1, max_seq_len),
                                      dtype="float32")
    # seq len + 2 for alphabet + eos + sos for input_data
    # seq len + 2 for alphabet + eos; neglect last token
    decoder_input_data = np.zeros((num_samples, seq_len+2, 3), dtype="float32")
    decoder_target_data = np.zeros((num_samples, seq_len+2, 3), dtype="float32")

    x, y = generate_dataset(num_samples, seq_len, num_repeat, repeat_dist,
                                num_tokens_rep, max_seq_len-1)

    if(verbose == 1):
        decode_seq(x, y, num_samples, seq_len)

    one_hot_encoding_label = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]])

    for i in range(num_samples):
        for seq in range(seq_len+1):
            # sos for decoder_input_data
            if seq == 0:
                decoder_input_data[i, seq] = one_hot_encoding_label[sos_decoder]
                encoder_input_data[i, seq] = x[i][seq]
            # generic case
            elif seq != seq_len:
                encoder_input_data[i, seq] = x[i][seq]
                # adjust for sos
                decoder_input_data[i, seq] = one_hot_encoding_label[y[i][seq-1]]
                # decoder target data lags decoder input by 1
                if (seq > 0):
                    decoder_target_data[i, seq-1] = decoder_input_data[i, seq]
            # append eos
            if seq == seq_len:
                encoder_input_data[i, seq] = eos_encoder
                decoder_input_data[i, seq] = one_hot_encoding_label[y[i][seq-1]]
                decoder_input_data[i, seq+1] = one_hot_encoding_label[eos_decoder]

                decoder_target_data[i, seq-1] = decoder_input_data[i, seq]
                decoder_target_data[i, seq] = one_hot_encoding_label[eos_decoder]

# train test split
num_train = 0.8*encoder_input_data.shape[0]
encoder_input_data_train = encoder_input_data[0:int(num_train)][:][:]
decoder_input_data_train = decoder_input_data[0:int(num_train)][:][:]
decoder_target_data_train = decoder_target_data[0:int(num_train)][:][:]

num_test = 0.2*encoder_input_data.shape[0]
encoder_input_data_test = encoder_input_data[int(num_test):][:][:]
decoder_input_data_test = decoder_input_data[int(num_test):][:][:]
decoder_target_data_test = decoder_target_data[int(num_test):][:][:]

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, 3))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# divide into training and test sets 80% and 20%

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data_train, decoder_input_data_train],
    decoder_target_data_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.3,
)
# Save model
#model.save("s2s")

# Define sampling models
# Restore the model and construct the encoder and decoder.
#model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

if input_seq == 'default':
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    for seq_index in range(20):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print("-")
        print("Input sentence:", input_texts[seq_index])
        print("Decoded sentence:", decoded_sentence)

else:
    reverse_target_char_index = [0,1,2,3]
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 3))
        # Populate the first character of target sequence with the start character.
        target_seq[0,0:] = one_hot_encoding_label[sos_decoder]
        #target_seq[0, 0, target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = str(reverse_target_char_index[sampled_token_index])
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == eos_decoder or len(
                    decoded_sentence) > seq_len:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 3))
            target_seq[0, 0, :] = one_hot_encoding_label[sampled_token_index]

            # Update states
            states_value = [h, c]
        return decoded_sentence

    y_pred = np.zeros((len(encoder_input_data_test), seq_len+2, 3), dtype="float32")

    one_hot_encoding_label = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    for seq_index in range(len(encoder_input_data_test)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data_test[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        val_one_hot = np.zeros((1,seq_len+2,3),dtype="float32")
        for num, val in enumerate(decoded_sentence):
            val_one_hot[0][num][:] = one_hot_encoding_label[int(val)]
        y_pred[seq_index:seq_index+1] = val_one_hot
        """
        print("-")
        print("Decoded sentence:", decoded_sentence)
        seq = []
        for num in range(seq_len):
            seq.append(one_hot_decoding(alphabet, input_seq[0][num][:]))
        print("Input sentence:", seq)
        """
    print("Balanced accuracy of test set")
    y_true = decoder_target_data_test.argmax(axis=2).ravel()
    y_est = y_pred.argmax(axis=2).ravel()
    print(balanced_accuracy_score(y_true, y_est))
    print(classification_report(y_true, y_est))




