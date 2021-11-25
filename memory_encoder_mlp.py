# https://keras.io/examples/nlp/lstm_seq2seq/

import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping
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
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import pandas as pd
import pickle
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# transformer block implementations
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, memory_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.memory_model = memory_model
        if (self.memory_model != "transformer_no_orthonormal"):
            self.maxlen = maxlen
        # add encoding only when orthonormal encoding is not used
        if (self.memory_model == "transformer_no_orthonormal"):
            self.token_emb = layers.Embedding(input_dim=vocab_size,
                                              output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        if (self.memory_model == "transformer_no_orthonormal"):
            maxlen = tf.shape(x)[-1]
        if (self.memory_model != "transformer_no_orthonormal"):
            positions = tf.range(start=0, limit=self.maxlen, delta=1)
        else:
            positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        if (self.memory_model == "transformer_no_orthonormal"):
            x = self.token_emb(x)
        return x + positions


def load_dataset(args):
    # df = pd.read_csv('/workspace/memory_clean/Memory/memory_retention_raw.csv',
    #                 usecols=['index', 'seq_len', 'seq', 'rep_token_first_pos',
    #                          'query_token', 'target_val'])
    if args.debug == 1:
        df = pd.read_csv(args.root_location + "memory_retention_raw_26.csv",
                         usecols=['index', 'seq_len', 'seq',
                                  'rep_token_first_pos',
                                  'query_token', 'target_val'])
    else:
        df = pd.read_csv(args.root_location + "memory_retention_raw.csv",
                         usecols=['index', 'seq_len', 'seq',
                                  'rep_token_first_pos',
                                  'query_token', 'target_val'])
    print(df.head())
    len_seq = df['seq_len'].to_numpy()
    raw_sequence = df['seq'].to_numpy()
    rep_token_first_pos = df['rep_token_first_pos'].to_numpy()
    token_rep = df['query_token'].to_numpy()
    target_y = df['target_val'].to_numpy()

    # read the pickle file
    if args.debug == 1:
        f = open(args.root_location + 'input_data_26.pkl', 'rb')
        orth_vectors = np.load(
            args.root_location + 'orthonormal_vectors_26.npy')
    else:
        f = open(args.root_location + 'input_data.pkl', 'rb')
        orth_vectors = np.load(
            args.root_location + 'orthonormal_vectors_512.npy')
    x = pickle.load(f)
    f.close()
    ip_sequence = np.load(args.root_location + 'raw_sequence.npy',
                          allow_pickle=True)
    num_samples = len(x)
    raw_sample_length = len(raw_sequence)
    print("Number of samples {}".format(num_samples))
    print("Number of samples in raw sequence {}".format(raw_sample_length))
    return x, num_samples, len_seq, token_rep, rep_token_first_pos, \
           ip_sequence, target_y, orth_vectors


"""
get the token id from the from the sequence, required for transformers
"""


def orthonormal_decode(dataset, orthonormal_vectors):
    seq_dataset = []
    for sequence in dataset:
        seq = []
        for token in sequence:
            a = np.matmul(token, orthonormal_vectors.T)
            idx = np.isclose(a, 1)
            id = np.where(idx == True)
            if np.size(id) == 0:
                token_id = 0
            else:
                token_id = id[0][0]
                # do not append eos
                if (token_id == 511):
                    break
            seq.append(token_id)
        seq_dataset.append(seq)

    return seq_dataset


"""
This function parses and pads the data
"""


def process_data(max_seq_len, latent_dim, padding, memory_model, num_samples, x,
                 raw_sequence):
    # separate out the input to the encoder and the mlp
    # mlp is fed the last one hot encoded input
    x_mlp = [0] * num_samples
    x_encoder = [0] * num_samples

    for iter, seq in enumerate(x):
        # seq[-1] - eos seq[-2] - query token seq[0:-2] - seq
        x_mlp[iter] = seq[-2]
        # all but the last one hot encoded sequence
        x_encoder[iter] = seq[0:-2]
        # eos
        x_encoder[iter].append(seq[-1])

    # seq len + 1 for alphabet + eos as orthonormal vectors are created with eos
    # max size of seq len is not max seq len - 1 for the actual sequence + 1 for eos
    encoder_input_data = np.zeros((num_samples, max_seq_len,
                                   latent_dim * 2), dtype="float32")

    mlp_input_data = np.zeros((num_samples, latent_dim * 2), dtype="float32")

    if padding == 'pre_padding':
        print("The shape of the encoder data is: " + str(
            encoder_input_data.shape))
        for i in range(num_samples):
            seq_len = len(x_encoder[i])

            for seq in range(seq_len):
                # fill the elements in encoder_input_data in the reverse order,
                # this ensures that zero padding is done before the sequence
                encoder_input_data[i, max_seq_len - seq_len + seq] = \
                    x_encoder[i][seq]
            mlp_input_data[i] = x_mlp[i]
    elif padding == 'post_padding':

        for i in range(num_samples):
            seq_len = len(x_encoder[i])
            for seq in range(seq_len):
                encoder_input_data[i, seq] = x_encoder[i][seq]
            mlp_input_data[i] = x_mlp[i]

    if memory_model == "transformer_no_orthonormal":
        # remove the query token - the last token in raw_sequence
        sequence_raw = []
        for seq in raw_sequence:
            sequence_raw.append(seq[:-1])
        raw_sequence_padded = keras.preprocessing.sequence.pad_sequences(
            sequence_raw, maxlen=max_seq_len - 1, value=0)
    else:
        raw_sequence_padded = raw_sequence

    return encoder_input_data, mlp_input_data, raw_sequence_padded


def define_nn_model(max_seq_len, memory_model, latent_dim, raw_seq_train,
                    raw_seq_val):
    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim * 2))
    query_input_node = keras.Input(shape=(latent_dim * 2))

    if memory_model == "lstm":
        # Define an input sequence and process it.
        main_sequence = keras.Input(shape=(None, latent_dim * 2))
        query_input_node = keras.Input(shape=(latent_dim * 2))

        """
        # encoder = Sequential()
        dense_layer = Sequential()
        encoder = keras.layers.LSTM(128, return_state=True)
        # encoder.add(keras.layers.Dense(512))
        encoder_outputs, state_h, state_c = encoder(main_sequence)
        # state_h, state_c = encoder(main_sequence)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = tf.concat((state_h, state_c), 1)
        dense_layer.add(keras.layers.Dense(768))
        dense_layer.add(keras.layers.Dense(512))
        encoder_states = dense_layer(encoder_states)
        """
        encoder_outputs, state_h, state_c = keras.layers.LSTM(128, return_state=True)(main_sequence)
        encoder_states = tf.concat((state_h, state_c), 1)


        lr = 0.0013378606854350151
        print("Encoder chosen is LSTM")
    elif memory_model == "RNN":
        # Define an input sequence and process it.
        main_sequence = keras.Input(shape=(None, latent_dim * 2))
        query_input_node = keras.Input(shape=(latent_dim * 2))
        """
        encoder = Sequential()
        encoder.add(keras.layers.SimpleRNN(256))
        encoder.add(keras.layers.Dense(1024, activation='relu'))
        encoder.add(keras.layers.Dense(512))
        encoder_output = encoder(main_sequence)
        encoder_states = encoder_output
        encoder.summary()
        """
        encoder_states = keras.layers.SimpleRNN(256)(main_sequence)
        #encoder_states = keras.layers.BatchNormalization()(encoder_states)
        #encoder_states = keras.layers.Dropout(0.2)(encoder_states)
        #encoder_states = keras.layers.Dense(1024, activation='relu')(encoder_states)
        #encoder_states = keras.layers.BatchNormalization()(encoder_states)
        #encoder_states = keras.layers.Dropout(0.2)(encoder_states)
        #encoder_states = keras.layers.Dense(512)(encoder_states)

        print("Encoder chosen is simple RNN")
        print("Shape of the encoder output is: " + str(encoder_states))
        lr = 1.0465692011515144e-05
    elif memory_model == "CNN":
        input_shape = (max_seq_len, latent_dim * 2)
        main_sequence = keras.Input(shape=(None, latent_dim * 2))
        query_input_node = keras.Input(shape=(latent_dim * 2))
        """
        encoder = Sequential()
        # there are 256 different channels and each channel 7 tokens are taken at once, and convolution is performed
        # dimesion of input is max_seq_len(100)*latent_dim*2(512) so after convolution the output size is max_seq_len because padding is same
        # then padding must be such that the max value of 50 outputs are taken, so each filter has 2 outputs for max seq size = 100
        # so total outputs = latent_dim(256)*2 = 512; since output is concatenated with token make sure that the dimensions are same
        encoder.add(
            keras.layers.Conv1D(filters=128, kernel_size=3, padding='same',
                                activation='relu', input_shape=input_shape))
        encoder.add(
            keras.layers.Conv1D(filters=256, kernel_size=3, padding='same',
                                strides=2, activation='relu'))
        encoder.add(
            keras.layers.Conv1D(filters=512, kernel_size=3, padding='same',
                                strides=2, activation='relu'))

        # encoder.add(keras.layers.Dropout(0.3))
        encoder.add(keras.layers.GlobalMaxPooling1D())
        encoder.add(keras.layers.Dropout(0.3))
        # flatten makes the shape as [None, None]
        # encoder.add(Flatten())
        # encoder.add(keras.layers.Reshape((latent_dim*2,)))
        # encoder.add(Flatten())
        encoder.add(keras.layers.Dense(latent_dim * 2))

        encoder.summary()
        encoder_output = encoder(main_sequence)
        encoder_states = encoder_output
        """
        encoder_states = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same',
                            activation='relu', input_shape=input_shape)(main_sequence)
        encoder_states = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same',
                            strides=2, activation='relu')(encoder_states)
        encoder_states = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same',
                            strides=2, activation='relu')(encoder_states)
        encoder_states = keras.layers.GlobalMaxPooling1D()(encoder_states)
        #encoder_states = keras.layers.BatchNormalization()(encoder_states)
        #encoder_states = keras.layers.Dropout(0.2)(encoder_states)
        #encoder_states = keras.layers.Dense(latent_dim * 2)(encoder_states)

        print("Encoder chosen is CNN")
        print("Shape of the encoder output is: " + str(encoder_states))
        # lr = 0.00012691763008376296
        lr = 7.201800744529144e-05
    elif memory_model == "transformer":

        embed_dim = 32  # Embedding size for each token
        num_heads = 10  # Number of attention heads
        ff_dim = 8  # Hidden layer size in feed forward network inside transformer
        maxlen = max_seq_len
        vocab_size = max_seq_len
        main_sequence = keras.Input(shape=(None, latent_dim * 2))
        query_input_node = keras.Input(shape=(latent_dim * 2))
        # inputs = layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size,
                                                    embed_dim, memory_model)
        x = embedding_layer(main_sequence)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        encoder_output = layers.Dense(latent_dim * 2)(x)
        encoder_states = encoder_output
        print("Shape of the encoder output is: " + str(encoder_states))
    elif memory_model == "transformer_no_orthonormal":
        embed_dim = 32  # Embedding size for each token
        num_heads = 10  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        maxlen = max_seq_len
        vocab_size = maxlen
        main_sequence = layers.Input(shape=(maxlen - 1,))
        query_input_node = keras.Input(shape=(latent_dim * 2))

        # max length is 99 - do not restrict number of tokens; doesnt include eos
        # vocab_size is also 100 as there are 100 unique tokens
        embedding_layer = TokenAndPositionEmbedding(maxlen - 1, vocab_size,
                                                    embed_dim, memory_model)
        x = embedding_layer(main_sequence)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(latent_dim * 2)(x)
        encoder_states = outputs

        # encoder_input_data_train/test must now be decoded to have a list of token_ids
        # encoder_input_data_train = np.array(orthonormal_decode(encoder_input_data_train, orthonormal_vectors))
        # encoder_input_data_test = np.array(orthonormal_decode(encoder_input_data_test, orthonormal_vectors))
        encoder_input_train = raw_seq_train
        encoder_input_val = raw_seq_val

    # query_encoder = Sequential()
    # query_ip_shape = query_train.shape[1]
    # query_encoder.add(Dense(latent_dim*2, input_shape=(query_ip_shape,), activation='relu'))
    # query_encoded_op = query_encoder(query_input_node)

    num_classes = 2
    input_shape = encoder_states.shape[1]

    #concatenated_output = tf.reshape(
        #tf.reduce_sum(encoder_states * query_input_node, axis=1), (-1, 1))

    concatenated_output_shape = 1  # (latent_dim*4)+1
    print("The concatenated input shape is: " + str(concatenated_output_shape))

    y = keras.layers.Concatenate(axis=1)([encoder_states, query_input_node])
    #y = keras.layers.Dense(256, activation=keras.activations.relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dropout(0.2)(y)
    y = keras.layers.Dense(768, activation=keras.activations.relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dropout(0.2)(y)
    y = keras.layers.Dense(512, activation=keras.activations.relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dropout(0.2)(y)
    similarity_output = keras.layers.Dense(1, activation='sigmoid')(y)


    #similarity_output = tf.reshape(
    #    tf.reduce_sum(encoder_states * query_input_node, axis=1), (-1, 1))
    # construct another model to learn the similarities between the encoded
    # input and the query vector
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([main_sequence, query_input_node], similarity_output)
    model.summary()


    model.compile(
        #optimizer=RMSprop(learning_rate=1e-3),
        optimizer=Adam(learning_rate=1e-3),
        #loss=keras.losses.BinaryCrossentropy(from_logits=True),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    return model


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_acc = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.test_acc.append(acc)
        self.test_loss.append(loss)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def train_model(batch_size, epochs, memory_model, model,
                encoder_input_train, encoder_input_val, query_input_train,
                query_input_val, target_train, target_val, checkpoint_filepath):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True
    )

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.9

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # test train split on the train dataset

    history = model.fit(
        [encoder_input_train, query_input_train],
        target_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([encoder_input_val, query_input_val], target_val),
        callbacks=[model_checkpoint_callback, callback]  # ,
        #           TestCallback((encoder_input_val, query_input_val))]
    )

    print("Number of epochs run: " + str(len(history.history["loss"])))

    # save the validation and test accuracy and loss for further plotting
    # np.save('test_accuracy_' + str(memory_model), np.array(test_acc))
    # np.save('test_loss_' + str(memory_model), np.array(test_loss))
    np.save('val_accuracy_' + str(memory_model),
            np.array(history.history["val_accuracy"]))
    np.save('val_loss_' + str(memory_model),
            np.array(history.history["val_loss"]))

    return


def predict_model(model, target_val, encoder_input_val,
                  query_input_val):

    y_test = model.predict([encoder_input_val, query_input_val])
    #y_test = sigmoid(y_test)
    # y_pred = np.argmax(y_test, axis=1)
    # for the kernel functions you would need values which are not 0 or 1
    y_pred_binary = np.array([1 if y > 0.5 else 0 for y in y_test])
    y_pred_continuous = y_test
    return y_pred_binary, y_pred_continuous


def compute_save_metrics(max_seq_len, memory_model, y_true, y_pred,
                         sequence_length_val,
                         rep_token_pos_val):
    # total balanced accuracy accross the entire test dataset
    #y_pred = sigmoid(y_pred)
    y_pred = np.array([1 if y > 0.5 else 0 for y in y_pred])
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Find the balanced accuracy accross different sequence length
    sequence_len_arr = np.array(sequence_length_val)
    # balanced_acc_seq_len of 0 and 1 are meaningless
    balanced_acc_seq_len = np.zeros(
        shape=(max_seq_len, max_seq_len))  # [0]*(max_seq_len+1)

    dist_arr = []
    rep_first_token_test = np.array(rep_token_pos_val)
    # dist_test == max_seq_len means there were no repeats,
    # should be fine as we ignore entries with max len later on
    rep_token_test = np.where(rep_first_token_test == -1, max_seq_len,
                              rep_first_token_test)
    dist_test = np.subtract(sequence_len_arr, np.add(rep_token_test, 1))

    for seq_len in range(1, max_seq_len):
        # balanced_acc_seq_len.append([])
        for dist in range(0, seq_len):

            # get the indices of samples which have a particular sequence length
            seq_len_indices = np.where(
                (sequence_len_arr == seq_len) & (dist_test == dist))

            # splice y_true and y_pred based on the seq length
            y_true_seq_len = np.take(y_true, seq_len_indices[0])
            y_pred_seq_len = np.take(y_pred, seq_len_indices[0])

            #print("The number of sequences are: " + str(len(y_true_seq_len)))
            if len(y_true_seq_len) > 0:
                balanced_acc_seq_len[seq_len][dist] = balanced_accuracy_score(
                    y_true_seq_len, y_pred_seq_len)

            #print(
            #    "Balanced accuracy for seq len {} and dist {} is {}".format(
            #        seq_len,
            #        dist,
            #        balanced_acc_seq_len[
            #            seq_len][
            #            dist]))

    # save the balanced accuracy per seq len
    f = open('balanced_acc_seq_len_dist_' + memory_model + '.pkl', 'wb')
    pickle.dump(balanced_acc_seq_len, f, -1)
    f.close()

    return dist_test, balanced_accuracy


def sigmoid(x):
    z = np.exp(-1.0 * x)
    sig = 1.0 / (1.0 + z)
    return sig


def compute_optimal_tau(kern, avg_test_acc, y_true, y_pred, dist_test,
                        sequence_length_val):
    # difficulty = seq len; time elapsed since last review = dist; strength =
    # average accuracy.
    # normalize s and d by dividing by 100
    x = [((s * d * 1.0) / ((avg_test_acc+np.finfo(float).eps) * 100 * 100)) for s, d in
         zip(sequence_length_val, dist_test)]
    #test_accs = np.array(y_true.ravel()) & np.array(y_pred.ravel())
    #print(test_accs.shape)
    #test_accs = [0.1 if acc < 1. else 0.9 for acc in
    #             test_accs.squeeze().tolist()]
    # test accs now are continuous non-zero values

    test_accs = np.array(y_pred.ravel())
    test_accs = [0.001 if test_acc == 0 else test_acc for test_acc in test_accs]


    if kern == 'Gaussian':
        # throughout training - take average error
        # earlyon maybe a different model is better and maybe at the end a diff
        # model is good - good to capture
        # do this on the validation data

        # epochs1 - k use validation acc as strength of model
        # at teh end use test acc as strength of model
        # s and d normalize - s - 1 - 100  -> 0.01 - 1 d - 0.01 - 1
        # use validation acc instead of test acc - best validation acc
        # best epoch try all functions - both papers on val data
        # then do this for every epoch - val data
        # dont use test data to tune hyperparams
        # gaussian
        num = -1.0 * np.sum(np.log(test_accs))
        den = np.sum(np.power(x, 2))

    if kern == "Laplacian":
        num = -1.0 * np.sum(np.log(test_accs))
        den = np.sum(x)

    if kern == "Linear":
        num = np.sum(np.sum(np.subtract(1, test_accs)))
        den = np.sum(x)

    if kern == "Cosine":
        num = np.sum(np.arccos(np.subtract(np.multiply(2., test_accs), 1)))
        den = np.pi * np.sum(x)

    if kern == "Quadratic":
        num = np.sum(np.subtract(1.0, test_accs))
        den = np.sum(np.power(x, 2))

    if kern == "Secant":
        # num = np.sum(np.log(1. / np.sum(test_accs) + np.sqrt(
        #     1. / np.sum(np.subtract(np.power(test_accs, 2), 1.)))))
        num = np.sum(np.log(1. / np.sum(test_accs) + np.sqrt(
             np.subtract(np.sum((1. / np.power(test_accs, 2))), 1.))))
        den = np.sum(np.power(x, 2))

    tau = num * 1.0 / den
    return tau, test_accs


def compute_l2_loss(tau, kern, test_accs):
    if kern == 'Gaussian':
        print("computing l2 loss")
        f_gauss = np.exp(-1 * tau * np.sum(np.power(x, 2)))
        # test_acc b/w 0 and 1
        f_gauss_loss = np.mean(np.power((f_gauss - test_accs), 2))
        return f_gauss_loss

    if kern == "Laplacian":
        f_lap = np.exp(-1 * tau * np.sum(x))
        # test_acc b/w 0 and 1
        f_lap_loss = np.mean(np.power((f_lap - test_accs), 2))
        return f_lap_loss

    if kern == "Linear":
        f_lin = (1 - (1 * tau * np.sum(x)))
        f_lin_loss = np.mean(np.power((f_lin - test_accs), 2))
        return f_lin_loss

    if kern == "Cosine":
        f_cos = 1 / 2 * np.cos(tau * np.sum(x) * np.pi)
        f_cos_loss = np.mean(np.power((f_cos - test_accs), 2))
        return f_cos_loss

    if kern == "Quadratic":
        f_qua = 1 - tau * np.sum(np.power(x, 2))
        f_qua_loss = np.mean(np.power((f_qua - test_accs), 2))
        return f_qua_loss

    if kern == "Secant":
        f_sec = 2 * 1.0 / (np.exp(-1 * tau * np.sum(np.power(x, 2))) + np.exp(
            1 * tau * np.sum(np.power(x, 2))))
        f_sec_loss = np.mean(np.power((f_sec - test_accs), 2))
        return f_sec_loss


def compute_loss_forgetting_functions(forgetting_function, avg_test_acc,
                                      dist_test, sequence_length_val, test_accs):

    # difficulty = seq len; time elapsed since last review = dist; strength =
    # average accuracy.
    # exp(-seq_len*intervening_tokens/avg_test_acc)

    if forgetting_function == 'diff_dist_strength':
        x = [((s * d * 1.0) / ((avg_test_acc+np.finfo(float).eps) * 100 * 100)) for s, d in
             zip(sequence_length_val, dist_test)]
        x = np.array(x)
        f_diff_dist_strength = np.exp(-x)
        f_diff_dist_strength_loss = np.mean(np.power
                                            ((f_diff_dist_strength - test_accs), 2))
        return f_diff_dist_strength_loss

    # exp(-seq_len*intervening_tokens)
    elif forgetting_function == 'diff_dist':
        x = [((s * d * 1.0) / (100 * 100)) for s, d in
             zip(sequence_length_val, dist_test)]
        x = np.array(x)
        f_diff_dist = np.exp(-x)
        f_diff_dist_loss = np.mean(np.power
                                   ((f_diff_dist - test_accs), 2))
        return f_diff_dist_loss

    # exp(-seq_len/avg_test_acc)
    elif forgetting_function == 'diff_strength':
        x = [((s * 1.0) / ((avg_test_acc+np.finfo(float).eps) * 100 * 100)) for s, d in
             zip(sequence_length_val, dist_test)]
        x = np.array(x)
        f_diff_strength = np.exp(-x)
        f_diff_strength_loss = np.mean(np.power
                                       ((f_diff_strength - test_accs), 2))
        return f_diff_strength_loss

def kernel_matching(y_true, y_pred, dist_test, sequence_length_val,
                    y_pred_binary_pos_samples):
    kernels = ['Gaussian', 'Laplacian', 'Linear', 'Cosine', 'Quadratic',
               'Secant']

    avg_test_acc = balanced_accuracy_score(y_true, y_pred_binary_pos_samples)
    print("computing optimal tau")
    kern_loss = []
    tau_kernels = []
    exp_forgetting_function_loss = []
    # compute x - seq_len*dist

    for kern in kernels:
        print("Kernel type is {}".format(kern))

        tau, test_accs = compute_optimal_tau(kern, avg_test_acc, y_true, y_pred,
                                             dist_test, sequence_length_val)
        tau_kernels.append(tau)
        print("optimal value of tau is {}".format(tau))
        l2_loss = compute_l2_loss(tau, kern, test_accs)
        print("L2 loss for kernel {} is {}".format(kern, l2_loss))
        kern_loss.append(l2_loss)

    # compute l2 loss for functions from Reddy et al paper
    exp_forgetting_functions = ['diff_dist_strength', 'diff_dist',
                                'diff_strength']

    #test_accs = np.array(y_true.ravel()) & np.array(y_pred.ravel())
    test_accs = np.array(y_pred.ravel())
    for exp_forgetting_function in exp_forgetting_functions:
        exp_forgetting_l2_loss = compute_loss_forgetting_functions(
            exp_forgetting_function, avg_test_acc, dist_test, sequence_length_val,
            test_accs)
        exp_forgetting_function_loss.append(exp_forgetting_l2_loss)


    # find the least loss
    min_index = kern_loss.index(min(kern_loss))
    print("The best kernel is {}".format(kernels[min_index]))
    print("the value of the loss is {}".format(min(kern_loss)))

    min_index_exp_forgetting_function = \
        exp_forgetting_function_loss.index(min(exp_forgetting_function_loss))
    print("The best forgetting function is {}".format(exp_forgetting_functions[min_index_exp_forgetting_function]))

    print("the value of the loss is {}".format(min(exp_forgetting_function_loss)))
    return kernels[min_index], tau_kernels[min_index]


def main(args):
    if args.debug == 1:
        args.root_location = \
            '/Users/sherin/Documents/research/server_version_memory/Memory/'

    else:
        args.root_location = '/workspace/memory_clean/Memory/'
    print("Loading the dataset")
    x, num_samples, sequence_len, token_repeated, rep_token_first_pos, \
    raw_sequence, target_y, orth_vectors = load_dataset(args)

    print("processing the dataset")
    encoder_input_data, query_data, raw_sequence_padded = \
        process_data(args.max_seq_len, args.latent_dim, args.padding,
                     args.nn_model, num_samples,
                     x, raw_sequence)

    print("Creating train and test split")

    # train test split
    (encoder_input_data_train, encoder_input_data_test,
     query_train, query_test,
     target_y_train, target_y_test,
     sequence_len_train, sequence_len_test,
     token_repeated_train, token_repeated_test,
     rep_token_first_pos_train, rep_token_first_pos_test,
     raw_sequence_train, raw_sequence_test) = train_test_split(
        encoder_input_data,
        query_data,
        target_y,
        sequence_len,
        token_repeated,
        rep_token_first_pos,
        raw_sequence_padded,
        random_state=2,
        test_size=0.3)

    # train val split
    (encoder_input_train, encoder_input_val,
     query_input_train, query_input_val,
     target_train, target_val,
     sequence_length_train, sequence_length_val,
     token_rep_train, token_rep_val,
     rep_token_pos_train, rep_token_pos_val,
     raw_seq_train, raw_seq_val) = train_test_split(
        encoder_input_data_train,
        query_train,
        target_y_train,
        sequence_len_train,
        token_repeated_train,
        rep_token_first_pos_train,
        raw_sequence_train,
        random_state=2,
        test_size=0.3)

    print("The number of examples in the training data set is " + str(
        len(encoder_input_train)))
    print("The number of example in the test data set is " + str(
        len(encoder_input_data_test)))
    print("The number of example in the validation data set is " + str(
        len(encoder_input_val)))

    # define the neural network model
    print("defining the Neural Network")
    model = define_nn_model(args.max_seq_len, args.nn_model, args.latent_dim,
                            raw_seq_train,
                            raw_seq_val)

    # train and save the best model
    # adding params like epoch and val accuracy will save all the models
    checkpoint_filepath = 'best_model_' + str(args.nn_model)

    print("training the neural network")
    train_model(args.batch_size, args.epochs, args.nn_model, model,
                encoder_input_train, encoder_input_val, query_input_train,
                query_input_val, target_train, target_val, checkpoint_filepath)

    # load the best model after training is complete
    print("loading the best model")
    model = keras.models.load_model(checkpoint_filepath)

    # test the model on novel data
    print("predicting on novel inputs")
    y_pred_binary, y_pred_continuous = predict_model(model, target_val,
                                         encoder_input_val, query_input_val)

    # compute accuracy based on seq len and number of intervening tokens
    dist_test, balanced_accuracy = compute_save_metrics(args.max_seq_len,
                                                        args.nn_model,
                                                        target_val, y_pred_binary,
                                                        sequence_length_val,
                                                        rep_token_pos_val)

    print("The balanced accuracy is {}".format(balanced_accuracy))

    # we need to calculate p(recall) only for positive instances, whether the
    # model is able to recall a previously seen item or not, so remove the
    # negative instances : where the query token is not previously seen by the
    # model
    negative_samples = np.where(target_val == 0)


    target_val_pos_samples = np.delete(target_val, negative_samples[0])
    y_pred_pos_samples = np.delete(y_pred_continuous, negative_samples[0])
    y_pred_binary_pos_samples = np.delete(y_pred_binary, negative_samples[0])
    dist_pos_samples = np.delete(dist_test, negative_samples[0])
    seq_len_test_pos_samples = np.delete(sequence_length_val, negative_samples[0])


    # learn which kernel best models the test accuracy
    print("Finding the best kernel to model the test accuracy")
    kernel, tau = kernel_matching(target_val_pos_samples, y_pred_pos_samples,
                                  dist_pos_samples, seq_len_test_pos_samples,
                                  y_pred_binary_pos_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_model", default="lstm", type=str,
                        help="neural network model to be used")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of epochs to be run for")
    parser.add_argument("--batch_size", default=50, type=int,
                        help="Number of samples in one batch")
    parser.add_argument("--latent_dim", default=256, type=int,
                        help="size of the memory encoding")
    parser.add_argument("--padding", default="post_padding", type=str,
                        help="Type of padding, pre-padding or "
                             "post-padding")
    parser.add_argument("--max_seq_len", default=26, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--debug", type=int, default=1, help="is it debug")
    args = parser.parse_args()

    print(args)

    main(args)
