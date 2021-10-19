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
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
# dataset is fra.txt which is downloaded from http://www.manythings.org/anki/fra-eng.zip

# memory model can be lstm, rnn, or cnn
memory_model = "lstm"
#memory_model = "CNN"
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
#epochs = 200
epochs = 1
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

print("The number of examples in the training data set is " + str(len(encoder_input_data_train)))
print("The number of example in the test data set is " + str(len(encoder_input_data_test)))
# Define an input sequence and process it.
main_sequence = keras.Input(shape=(None, latent_dim*2))
query_input_node = keras.Input(shape=(latent_dim*2))

if(memory_model == "lstm"):
    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))
    #encoder = Sequential()
    dense_layer = Sequential()
    encoder = keras.layers.LSTM(128, return_state=True)
    #encoder.add(keras.layers.Dense(512))
    encoder_outputs, state_h, state_c = encoder(main_sequence)
    #state_h, state_c = encoder(main_sequence)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = tf.concat((state_h, state_c), 1)
    dense_layer.add(keras.layers.Dense(768))
    dense_layer.add(keras.layers.Dense(512))
    encoder_states = dense_layer(encoder_states)
    lr = 0.0013378606854350151
    print("Encoder chosen is LSTM")
elif(memory_model == "RNN"):
    # Define an input sequence and process it.
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))
    encoder = Sequential()
    encoder.add(keras.layers.SimpleRNN(256))
    encoder.add(keras.layers.Dense(1024, activation='relu'))
    encoder.add(keras.layers.Dense(512))
    encoder_output = encoder(main_sequence)
    encoder_states = encoder_output
    encoder.summary()
    
    print("Encoder chosen is simple RNN")
    print("Shape of the encoder output is: " + str(encoder_states))
    lr = 1.0465692011515144e-05
elif(memory_model == "CNN"):
    input_shape = (max_seq_len, latent_dim*2)
    main_sequence = keras.Input(shape=(None, latent_dim*2))
    query_input_node = keras.Input(shape=(latent_dim*2))
    encoder = Sequential()
    # there are 256 different channels and each channel 7 tokens are taken at once, and convolution is performed
    # dimesion of input is max_seq_len(100)*latent_dim*2(512) so after convolution the output size is max_seq_len because padding is same
    # then padding must be such that the max value of 50 outputs are taken, so each filter has 2 outputs for max seq size = 100
    # so total outputs = latent_dim(256)*2 = 512; since output is concatenated with token make sure that the dimensions are same
    encoder.add(keras.layers.Conv1D(filters = 128, kernel_size = 3, padding='same', activation='relu', input_shape=input_shape))
    encoder.add(keras.layers.Conv1D(filters = 256, kernel_size = 3, padding='same', strides = 2, activation='relu'))
    encoder.add(keras.layers.Conv1D(filters = 512, kernel_size = 3, padding='same', strides = 2, activation='relu'))
 

    #encoder.add(keras.layers.Dropout(0.3))
    encoder.add(keras.layers.GlobalMaxPooling1D())
    encoder.add(keras.layers.Dropout(0.3))
    # flatten makes the shape as [None, None]
    #encoder.add(Flatten())
    #encoder.add(keras.layers.Reshape((latent_dim*2,)))
    #encoder.add(Flatten())
    encoder.add(keras.layers.Dense(latent_dim*2))
    
    encoder.summary()
    encoder_output = encoder(main_sequence)
    encoder_states = encoder_output
    print("Encoder chosen is CNN")
    print("Shape of the encoder output is: " + str(encoder_states))
    #lr = 0.00012691763008376296
    lr = 7.201800744529144e-05
elif(memory_model == "transformer"):

    embed_dim = 32  # Embedding size for each token
    num_heads = 10  # Number of attention heads
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
    encoder_output = layers.Dense(latent_dim*2)(x)
    encoder_states = encoder_output
    print("Shape of the encoder output is: " + str(encoder_states))
elif (memory_model == "transformer_no_orthonormal"):
    embed_dim = 32  # Embedding size for each token
    num_heads = 10  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = max_seq_len
    vocab_size = maxlen
    main_sequence = layers.Input(shape=(maxlen-1,))
    query_input_node = keras.Input(shape=(latent_dim * 2))

    # max length is 99 - do not restrict number of tokens; doesnt include eos
    # vocab_size is also 100 as there are 100 unique tokens
    embedding_layer = TokenAndPositionEmbedding(maxlen-1, vocab_size, embed_dim)
    x = embedding_layer(main_sequence)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(latent_dim*2)(x)
    encoder_states = outputs

    # encoder_input_data_train/test must now be decoded to have a list of token_ids
    #encoder_input_data_train = np.array(orthonormal_decode(encoder_input_data_train))
    #encoder_input_data_test = np.array(orthonormal_decode(encoder_input_data_test))
    encoder_input_data_train = raw_sequence_train
    encoder_input_data_test = raw_sequence_test

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
model = keras.Model([main_sequence, query_input_node], concatenated_output)
model.summary()


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(
    optimizer=RMSprop(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"]
)

# early stopping
es_cb = EarlyStopping(monitor="val_loss", patience=100, verbose=1,
                      mode="min")
# adding params like epoch and val accuracy will save all the models
#checkpoint_filepath = "./models/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint_filepath = 'best_model_'+str(memory_model)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True
)

test_acc = []
test_loss = []
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        test_acc.append(acc)
        test_loss.append(loss)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

#y_mlp_binary_train = to_categorical(np.array(y_mlp_train), dtype="float32")

# debug
"""
def check(test,array):
    for idx, x in enumerate(array):
        a = np.matmul(test, array.T)
        idx = np.isclose(a, 1)
        id = np.where(idx == True)
        if np.size(id) == 0:
            break
        else:
            return id[0][0]

    return -1


print("the sequence is ")
for token in encoder_input_data_train[0]:
    #token_id = np.dot(orthonormal_vectors.T, orthonormal_vectors)
    token_id = check(token, orthonormal_vectors)
    if(token_id != -1):
        print(token_id)
    else:
        print("Token id not found")

print("the query id is")
query_id = check(query_train[0], orthonormal_vectors)
if (query_id != -1):
    print(query_id)
else:
    print("Token id not found")

print("the mlp output value is ")
print(y_mlp_binary_train[0])
exit()
"""
history = model.fit(
    [encoder_input_data_train, query_train],
    y_mlp_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.3,
    callbacks=[model_checkpoint_callback, TestCallback((encoder_input_data_test, mlp_input_data_test))]
)

print("Number of epochs run: " + str(len(history.history["loss"])))

# save the validation and test accuracy and loss for further plotting
np.save('test_accuracy_'+str(memory_model), np.array(test_acc))
np.save('test_loss_' + str(memory_model), np.array(test_loss))
np.save('val_accuracy_'+str(memory_model), np.array(history.history["val_accuracy"]))
np.save('val_loss_'+str(memory_model), np.array(history.history["val_loss"]))


# load the best model which is saved
#model = keras.models.load_model(checkpoint_filepath)
#print("loaded the best model")
#y_true = np.array(y_mlp_test, dtype="float32")
y_true = y_mlp_test
# test results
print("starting the prediction")
y_test = model.predict([encoder_input_data_test, mlp_input_data_test])
#y_pred = np.argmax(y_test, axis=1)
y_pred = y_test>0.5
"""
print("The true values are:")
print(y_true)
print("The predicted values are: ")
print(y_pred)
"""

# total balanced accuracy accross the entire test dataset
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(balanced_accuracy)

# Find the balanced accuracy accross different sequence length
sequence_len_arr = np.array(sequence_len_test)
# balanced_acc_seq_len of 0 and 1 are meaningless
balanced_acc_seq_len = np.zeros(shape=(max_seq_len, max_seq_len)) #[0]*(max_seq_len+1)

dist_arr = []
rep_first_token_test = np.array(rep_token_first_pos_test)
# dist_test == max_seq_len means there were no repeats, should be fine as we ignore
# entries with max len later on
rep_token_test = np.where(rep_first_token_test == -1, max_seq_len, rep_first_token_test)
dist_test = np.subtract(sequence_len_arr, np.add(rep_token_test, 1))

print("computing optimal tau")
mean_loss = []
# compute x - seq_len*dist
avg_test_acc = balanced_accuracy_score(y_true, y_pred)
x = [((s*d*1.0)/avg_test_acc) for s, d in zip(sequence_len_test, dist_test)]
test_accs = np.array(y_true) & np.array(y_pred)
#print(test_accs.squeeze().tolist())
test_accs = [0.1 if acc <1. else 0.9 for acc in test_accs.squeeze().tolist()]

# gaussian
num = -1.0*np.sum(np.log(test_accs))
den = np.sum(np.pow(x,2))
tau = num*1.0/den

# laplacian


# compute l2 loss
print("computing l2 loss")
f_gauss = np.exp(-1*tau*np.sum(np.pow(x,2)))
f_gauss_loss = np.mean(np.pow((f_gauss - test_accs), 2))
mean_loss.append(f_gauss_loss)

kernels = ['Gaussian', 'Laplacian', 'Linear', 'Cosine', 'Quadratic', 'Secant']
min_val = min(mean_loss)
min_index = mean_loss.index(min_val)

print("The best function is ")
print(kernels[min_index])


# save mean loss
np.save('mean_loss', mean_loss)




# assume gaussian kernel


for seq_len in range(1,max_seq_len):
    #balanced_acc_seq_len.append([])
    for dist in range(0, seq_len):

    # get the indices of samples which have a particular sequence length
        seq_len_indices = np.where((sequence_len_arr == seq_len) & (dist_test == dist))

        # splice y_true and y_pred based on the seq length
        y_true_seq_len = np.take(y_true, seq_len_indices[0])
        y_pred_seq_len = np.take(y_pred, seq_len_indices[0])

        print("The number of sequences are: " + str(len(y_true_seq_len)))
        if len(y_true_seq_len) > 0:
            balanced_acc_seq_len[seq_len][dist] = balanced_accuracy_score(y_true_seq_len,y_pred_seq_len)

        print("Balanced accuracy for seq len {} and dist {} is {}".format(seq_len, dist, balanced_acc_seq_len[seq_len][dist]))


# save the balanced accuracy per seq len
f = open('balanced_acc_seq_len_dist_'+memory_model+'.pkl', 'wb')
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




