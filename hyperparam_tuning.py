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
import optuna

from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import RMSprop

# memory model can be lstm, rnn, or cnn
#memory_model = "lstm"
memory_model = "CNN"
#memory_model = "RNN"
#memory_model = "transformer"
#memory_model = 'transformer_no_orthonormal'

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
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        if (memory_model != "transformer_no_orthonormal"):
            self.maxlen = maxlen
        # add encoding only when orthonormal encoding is not used
        if(memory_model == "transformer_no_orthonormal"):
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        if (memory_model == "transformer_no_orthonormal"):
            maxlen = tf.shape(x)[-1]
        if (memory_model != "transformer_no_orthonormal"):
            positions = tf.range(start=0, limit=self.maxlen, delta=1)
        else:
            positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        if (memory_model == "transformer_no_orthonormal"):
            x = self.token_emb(x)
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

df = pd.read_csv('/workspace/Memory/memory_retention_raw.csv', usecols=['index', 'seq_len', 'seq', 'rep_token_first_pos', 'query_token', 'target_val'])
print(df.head())

sequence_len = df['seq_len'].to_numpy()
raw_sequence = df['seq'].to_numpy()
rep_token_first_pos = df['rep_token_first_pos'].to_numpy()
token_repeated = df['query_token'].to_numpy()
y_mlp = df['target_val'].to_numpy()
num_samples = len(raw_sequence)


# read the pickle file
f = open('/workspace/Memory/input_data.pkl', 'rb')
x = pickle.load(f)
f.close()

orthonormal_vectors = np.load('/workspace/Memory/orthonormal_vectors_512.npy')

raw_sequence = np.load('raw_sequence.npy', allow_pickle=True)

"""
get the token id from the from the sequence, required for transformers
"""
def orthonormal_decode(dataset):
    seq_dataset = []
    for sequence in dataset:
        seq = []
        for token in sequence:
            a = np.matmul(token, orthonormal_vectors.T)
            idx = np.isclose(a,1)
            id = np.where(idx == True)
            if np.size(id) == 0:
                token_id = 0
            else:
                token_id = id[0][0]
                # do not append eos
                if(token_id == 511):
                    break
            seq.append(token_id)
        seq_dataset.append(seq)

    return seq_dataset





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

if(memory_model == "transformer_no_orthonormal"):
    # remove the query token - the last token in raw_sequence
    sequence_raw = []
    for seq in raw_sequence:
        sequence_raw.append(seq[:-1])
    raw_sequence_padded = keras.preprocessing.sequence.pad_sequences(sequence_raw, maxlen=max_seq_len-1, value = 0)
else:
    raw_sequence_padded = raw_sequence
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
 rep_token_first_pos_train, rep_token_first_pos_test,
 raw_sequence_train, raw_sequence_test) = train_test_split(encoder_input_data, mlp_input_data, y_mlp,
                                 sequence_len, token_repeated, rep_token_first_pos, raw_sequence_padded, random_state=2, test_size=0.3)



# validation and train data
(encoder_input_data_train, encoder_input_data_valid,
 query_train, mlp_input_data_valid,
 y_mlp_train, y_mlp_valid,
 sequence_len_train, sequence_len_valid,
 token_repeated_train, token_repeated_valid,
 rep_token_first_pos_train, rep_token_first_pos_valid) = train_test_split(encoder_input_data_train,
                                                           query_train,
                                                           y_mlp_train,
                                                           sequence_len_train,
                                                           token_repeated_train,
                                                           rep_token_first_pos_train,
                                                           random_state=2,
                                                           test_size=0.3)


print("The number of examples in the training data set is " + str(
    len(encoder_input_data_train)))
print("The number of example in the test data set is " + str(
    len(encoder_input_data_test)))
# Define an input sequence and process it.
main_sequence = keras.Input(shape=(None, latent_dim * 2))
query_input_node = keras.Input(shape=(latent_dim * 2))

def objective(trial):
    clear_session()

    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim * 2))
    query_input_node = keras.Input(shape=(latent_dim * 2))

    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(main_sequence)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = tf.concat((state_h, state_c), 1)

    num_classes = 2
    input_shape = encoder_states.shape[1]

    concatenated_output = tf.reshape(
        tf.reduce_sum(encoder_states * query_input_node, axis=1), (-1, 1))

    concatenated_output_shape = 1  # (latent_dim*4)+1
    print("The concatenated input shape is: " + str(concatenated_output_shape))

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([main_sequence, query_input_node], concatenated_output)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model.compile(
        optimizer=RMSprop(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        [encoder_input_data_train, query_train],
        y_mlp_train,
        validation_data = ([encoder_input_data_valid, mlp_input_data_valid], y_mlp_valid),
        batch_size=batch_size,
        epochs=epochs,
        verbose = False,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate([encoder_input_data_valid, mlp_input_data_valid], y_mlp_valid, verbose=0)
    # returns loss and metrics - accuracy
    return score[1]

def objective_cnn(trial):
    clear_session()

    input_shape =  (max_seq_len, latent_dim*2)
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))

    encoder = keras.Sequential()
    #encoder.add(keras.layers.Conv1D(filters = latent_dim, kernel_size = 7, padding = "same", activation='relu', input_shape = input_shape))
    #encoder.add(MaxPooling1D(pool_size=50))

    #encoder.add(keras.layers.Reshape((latent_dim*2,)))
    
    # before feeding the input reshape it to be 27
    encoder.add(keras.layers.Conv1D(filters = trial.suggest_categorical("filters", [5, 10, 16, 32, 64]), kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7]), strides=trial.suggest_categorical("strides", [1, 2]), padding = "same", activation='relu', input_shape = input_shape))
    encoder.add(Flatten())
    encoder.add(keras.layers.Dense(latent_dim*2))


    encoder_output = encoder(main_sequence)
    encoder_states = encoder_output

    num_classes = 2
    input_shape = encoder_states.shape[1]

    concatenated_output = tf.reshape(
        tf.reduce_sum(encoder_states * query_input_node, axis=1), (-1, 1))

    concatenated_output_shape = 1  # (latent_dim*4)+1
    print("The concatenated input shape is: " + str(concatenated_output_shape))

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([main_sequence, query_input_node], concatenated_output)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model.compile(
        optimizer=RMSprop(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(
        [encoder_input_data_train, query_train],
        y_mlp_train,
        validation_data = ([encoder_input_data_valid, mlp_input_data_valid], y_mlp_valid),
        batch_size=batch_size,
        epochs=epochs,
        verbose = False,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate([encoder_input_data_valid, mlp_input_data_valid], y_mlp_valid, verbose=0)
    # returns loss and metrics - accuracy
    return score[1]



if __name__ == "__main__":
    # maximize accuracy
    study = optuna.create_study(direction="maximize")
    if(memory_model == 'lstm'):
        study.optimize(objective, n_trials=10)
    elif(memory_model == 'CNN'):
        study.optimize(objective_cnn, n_trials=50)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_results = study.trials_dataframe()
    df_results.to_pickle('df_optuna_results' + memory_model + '.pkl')
    df_results.to_csv('df_optuna_results' + memory_model + '.csv')


