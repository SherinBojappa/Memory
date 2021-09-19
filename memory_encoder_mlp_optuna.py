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
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
#from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import pandas as pd
import pickle
# dataset is fra.txt which is downloaded from http://www.manythings.org/anki/fra-eng.zip

# transformer block implementations
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate =0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim = embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
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
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        #self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        #maxlen = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        #x = self.token_emb(x)
        return x + positions

batch_size = 50  # Batch size for training.
#batch_size = 5
#epochs = 5  # Number of epochs to train for.
epochs = 30
latent_dim = 256  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
data_path = "fra.txt"
input_seq = 'synthetic'
seq_len = 4
num_repeat = 1
num_tokens_rep = 1
max_seq_len = 100
eos_encoder = np.zeros(max_seq_len+1)
eos_encoder[0] = 1
eos_decoder = 2
sos_decoder = 3
verbose = 0
padding = 'pre_padding'
# memory model can be lstm, rnn, or cnn
#memory_model = "lstm"
#memory_model = "CNN"
#memory_model = "RNN"
memory_model = "transformer"

#x, y, y_mlp, raw_sequence, token_repeated, pos_first_token, sequence_len = generate_dataset(max_seq_len,
#                                                                      num_tokens_rep)

# load the orthonormal vectors
#orthonormal_vectors = np.load('orthonormal_vectors_26.npy')
#print("The size of orthonomal vectors is " + str(orthonormal_vectors.shape))
# read from csv file that has the sequence and metadata
df = pd.read_csv('/workspace/Memory/memory_retention_raw.csv', usecols=['index', 'seq_len', 'seq', 'rep_token_first_pos', 'query_token', 'target_val'])
print(df.head())

sequence_len = df['seq_len'].to_numpy()
raw_sequence = df['seq']
rep_token_first_pos = df['rep_token_first_pos']
token_repeated = df['query_token']
y_mlp = df['target_val'].to_numpy()
num_samples = len(raw_sequence)


# read the pickle file
f = open('/workspace/Memory/input_data.pkl', 'rb')
x = pickle.load(f)
f.close()

"""
capped_max_seq_len = 25
seq_id = np.where(sequence_len == capped_max_seq_len)
max_index = seq_id[0][-1]
#print("id is" + str(idx))
#max_index = idx[0][-1]
print("max_index is " + str(max_index))
x = x[0:max_index+1]
num_samples = len(x)
print("The total number of samples are: " + str(num_samples))
y_mlp = y_mlp[0:max_index+1]
sequence_len = sequence_len[0:max_index+1]
token_repeated = token_repeated[0:max_index+1]
rep_token_first_pos = rep_token_first_pos[0:max_index+1]
"""

# separate out the input to the encoder and the mlp
# mlp is fed the last one hot encoded input
x_mlp = [0]*num_samples
x_encoder = [0]*num_samples

for iter, seq in enumerate(x):
    # seq[-1] - eos seq[-2] - query token seq[0:-2] - seq
    x_mlp[iter] = seq[-2]
    # all but the last one hot encoded sequence
    x_encoder[iter] = seq[0:-2]
    #eos
    x_encoder[iter].append(seq[-1])


# seq len + 1 for alphabet + eos as orthonormal vectors are created with eos
# max size of seq len is not max seq len - 1 for the actual sequence + 1 for eos
encoder_input_data = np.zeros((num_samples, max_seq_len,
                               latent_dim*2), dtype="float32")

mlp_input_data = np.zeros((num_samples, latent_dim*2), dtype = "float32")

if(padding == 'pre_padding'):
    print("The shape of the encoder data is: " + str(encoder_input_data.shape))
    for i in range(num_samples):
        seq_len = len(x_encoder[i])
        
        for seq in range(seq_len):
            # fill the elements in encoder_input_data in the reverse order,
            # this ensures that zero padding is done before the sequence
            encoder_input_data[i, max_seq_len - seq_len + seq] = x_encoder[i][seq]
        mlp_input_data[i] = x_mlp[i]
elif(padding == 'post_padding'):

    for i in range(num_samples):
        seq_len = len(x_encoder[i])
        for seq in range(seq_len):
            encoder_input_data[i, seq] = x_encoder[i][seq]
        mlp_input_data[i] = x_mlp[i]

"""
N = (num_samples//10000)*10000

# train test split
(encoder_input_data_train, encoder_input_data_test,
 query_train, mlp_input_data_test,
 y_mlp_train, y_mlp_test,
 sequence_len_train, sequence_len_test,
 token_repeated_train, token_repeated_test,
 rep_token_first_pos_train, rep_token_first_pos_test) = train_test_split(encoder_input_data[:N], mlp_input_data[:N], y_mlp[:N],
                                 sequence_len[:N], token_repeated[:N], rep_token_first_pos[:N], random_state=2, test_size=0.3)
"""

# train test split
(encoder_input_data_train, encoder_input_data_test,
 query_train, mlp_input_data_test,
 y_mlp_train, y_mlp_test,
 sequence_len_train, sequence_len_test,
 token_repeated_train, token_repeated_test,
 rep_token_first_pos_train, rep_token_first_pos_test) = train_test_split(encoder_input_data, mlp_input_data, y_mlp,
                                 sequence_len, token_repeated, rep_token_first_pos, random_state=2, test_size=0.3)

print("The number of examples in the training data set is " + str(len(encoder_input_data_train)))
print("The number of example in the test data set is " + str(len(encoder_input_data_test)))
# Define an input sequence and process it.
main_sequence = keras.Input(shape=(None, latent_dim*2))
query_input_node = keras.Input(shape=(latent_dim*2))

if(memory_model == "lstm"):
    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))

    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(main_sequence)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = tf.concat((state_h, state_c), 1)
    print("Encoder chosen is LSTM")
elif(memory_model == "RNN"):
    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))

    encoder = keras.layers.SimpleRNN(latent_dim*2, return_state=True)
    encoder_output, state = encoder(main_sequence)
    encoder_states = encoder_output
    print("Encoder chosen is simple RNN")
    print("Shape of the encoder output is: " + str(encoder_states))
elif(memory_model == "CNN"):
    input_shape = (max_seq_len, latent_dim*2)
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))
    encoder = Sequential()
    # there are 256 different channels and each channel 7 tokens are taken at once, and convolution is performed
    # dimesion of input is max_seq_len(100)*latent_dim*2(512) so after convolution the output size is max_seq_len because padding is same
    # then padding must be such that the max value of 50 outputs are taken, so each filter has 2 outputs for max seq size = 100
    # so total outputs = latent_dim(256)*2 = 512; since output is concatenated with token make sure that the dimensions are same
    encoder.add(keras.layers.Conv1D(filters = latent_dim, kernel_size = 7, padding='same', activation='relu', input_shape=input_shape))
    encoder.add(MaxPooling1D(pool_size=50))

    # flatten makes the shape as [None, None]
    #encoder.add(Flatten())
    encoder.add(keras.layers.Reshape((latent_dim*2,)))
    encoder_output = encoder(main_sequence)
    encoder_states = encoder_output
    print("Encoder chosen is CNN")
    print("Shape of the encoder output is: " + str(encoder_states))
elif(memory_model == "transformer"):

    embed_dim = latent_dim*2  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 8  # Hidden layer size in feed forward network inside transformer
    maxlen = max_seq_len
    #vocab_size = max_seq_len+1
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))
    #inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
    x = embedding_layer(main_sequence)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    encoder_output = layers.Dense(latent_dim*2, activation="softmax")(x)
    encoder_states = encoder_output
    print("Shape of the encoder output is: " + str(encoder_states))

#query_encoder = Sequential()
#query_ip_shape = query_train.shape[1]
#query_encoder.add(Dense(latent_dim*2, input_shape=(query_ip_shape,), activation='relu'))
#query_encoded_op = query_encoder(query_input_node)


num_classes=2
input_shape = encoder_states.shape[1]

#concatenated_input = tf.concat((encoder_states, query_encoded_op), 1)
#concatenated_input = tf.concat((encoder_states, query_encoded_op, tf.reshape(tf.reduce_sum(encoder_states*query_encoded_op, axis=1),(-1,1))), 1)
concatenated_output = tf.reshape(tf.reduce_sum(encoder_states*query_input_node, axis=1),(-1,1))
#concatenated_input = tf.concat((encoder_states, query_encoded_op, tf.matmul(encoder_states, tf.transpose(query_encoded_op))), 1)
#concatenated_input_shape = concatenated_input.shape[1]
#concatenated_input_shape = batch_size+ latent_dim*4
concatenated_output_shape = 1 #(latent_dim*4)+1
print("The concatenated input shape is: " + str(concatenated_output_shape))
#joint_mlp = Sequential()
#joint_mlp.add(Dense(num_classes, input_shape = (concatenated_input_shape,), activation='softmax'))
#from tensorflow.keras.layers.normalization import BatchNormalization
#joint_mlp.add(BatchNormalization())
#joint_mlp.add(Dense(100, input_shape=(concatenated_input_shape,), activation='relu')) #, activation='relu'))
#joint_mlp.add(BatchNormalization())
#joint_mlp.add(Activation('relu'))
#joint_mlp.add(Dropout(0.2))
#joint_mlp.add(Dense(1000, activation = 'relu'))
#joint_mlp.add(Dense(50, activation = 'relu'))
#joint_mlp.add(Dense(num_classes, activation='softmax'))

#joint_mlp_output = joint_mlp(concatenated_input)
# divide into training and test sets 80% and 20%

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
def create_model(trial):
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))

    encoder = keras.layers.SimpleRNN(latent_dim*2, return_state=True)
    encoder_output, state = encoder(main_sequence)
    encoder_states = encoder_output
    print("Encoder chosen is simple RNN")
    print("Shape of the encoder output is: " + str(encoder_states))

    num_classes=2
    input_shape = encoder_states.shape[1]

    concatenated_output = tf.reshape(tf.reduce_sum(encoder_states*query_input_node, axis=1),(-1,1))
    concatenated_output_shape = 1 #(latent_dim*4)+1
    print("The concatenated input shape is: " + str(concatenated_output_shape))

    model = keras.Model([main_sequence, query_input_node], concatenated_output)
    #model = Sequential()
    #model.add(keras.layers.SimpleRNN(latent_dim*2, return_state=True))
    #dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
    #model.add(keras.layers.Dropout(rate=dropout))
    #model.add(keras.layers.Reshape(tf.reduce_sum(encoder_states*query_input_node, axis=1),(-1,1))))
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    opt = trial.suggest_categorical("optimizer", ['SGD', 'Adam', 'RMSprop', 'Adagrad'])
    model.compile(
            loss = "binary_crossentropy",
            optimizer = opt(lr=lr),
            metrics=["accuracy"],)
    return model


def objective(trial):
    keras.backend.clear_session()

    model = create_model(trial)

    history = model.fit(
        [encoder_input_data_train, query_train],
        y_mlp_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.3,
        callbacks=[KerasPruningCallback(trial, "val_accuracy")],
        verbose=1
)

    score = model.evaluate()
    return score[1]

print("Number of epochs run: " + str(len(history.history["loss"])))


#y_true = np.array(y_mlp_test, dtype="float32")
y_true = y_mlp_test
# test results
y_test = model.predict([encoder_input_data_test, mlp_input_data_test])
#y_pred = np.argmax(y_test, axis=1)
y_pred = y_test>0.5
print("The true values are:")
print(y_true)
print("The predicted values are: ")
print(y_pred)

# total balanced accuracy accross the entire test dataset
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(balanced_accuracy)

# Find the balanced accuracy accross different sequence length
sequence_len_arr = np.array(sequence_len_test)
# balanced_acc_seq_len of 0 and 1 are meaningless
balanced_acc_seq_len = [0]*(max_seq_len+1)

for seq_len in range(1,max_seq_len):
    seq_len_indices = []
    # get the indices of samples which have a particular sequence length
    seq_len_indices = np.where(sequence_len_arr == seq_len)
    # splice y_true and y_pred based on the seq length
    y_true_seq_len = np.take(y_true, seq_len_indices[0])
    y_pred_seq_len = np.take(y_pred, seq_len_indices[0])

    print("The number of sequences are: " + str(len(y_true_seq_len)))
    balanced_acc_seq_len[seq_len] = balanced_accuracy_score(y_true_seq_len,
                                                            y_pred_seq_len)
    print("Balanced accuracy for seq len {} is {}".format(seq_len, balanced_acc_seq_len[seq_len]))

# save the balanced accuracy per seq len
f = open('balanced_acc_seq_len_'+memory_model+'.pkl', 'wb')
pickle.dump(balanced_acc_seq_len, f, -1)
f.close()

"""
# plot the balanced accuracy per sequence length
x = np.arange(0,max_seq_len+1)
plt.title("Balanced accuracy versus sequence length")
plt.xlabel("sequence length")
plt.ylabel("balanced accuracy")
plt.plot(x, balanced_acc_seq_len)
plt.show()
"""




