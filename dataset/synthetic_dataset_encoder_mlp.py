"""
Generation of synthetic dataset to evaluate memory retention in neural models
"""
import random
from string import ascii_lowercase
from random import choice
from random import randrange
from random import shuffle
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

# input data
# one hot encoded sequence with eos
x = list()
# label
y = list()
y_mlp = list()
token_repeated = list()
pos_first_token = list()
sequence_len = list()
max_seq_len = 26
# ignore entries 0,1 - seq length is one based with min=2 and max=max_seq_len
samples_seq = [0] * (max_seq_len+1)
eos_seq_ip = [0] * (max_seq_len + 1)
eos_seq_ip[-1] = 1
seq_dict = {}

eos_decoder = 2
num_instances_per_seq_len = 5000

alphabet = 'abcdefghijklmnopqrstuvwxyz'
num_to_letter = {}
num_to_letter[max_seq_len] = "eos"
for num, letter in enumerate(alphabet):
    num_to_letter[num] = letter

def one_hot_decoding(alphabet, one_hot_ip):

    seq = list()
    pos = 0
    letter_assigned = 0
    for val in one_hot_ip:
        if val == 1:
            letter = num_to_letter[pos]
            letter_assigned = 1
            break
        pos = pos+1

    # numpy arrays are sized to the max input so you can get all 0s as well
    if(letter_assigned == 0):
        letter = 'past end of sequence'

    seq.append(letter)
    return seq


def one_hot_encoding(sequence, letter_to_num):
    one_hot_encoded = list()
    for num, letter in enumerate(sequence):
        one_hot_list = [0 for i in range(len(alphabet))]
        one_hot_list[letter_to_num[letter]] = 1
        one_hot_encoded.append(one_hot_list)

    return one_hot_encoded


def generate_labels(sequence):
    new_list = list()
    seq_len = len(sequence)
    label = [0]*(seq_len+1)
    for num, letter in enumerate(sequence):
        #print(num, letter)
        if letter in new_list:
            label[num] = 1

        new_list.append(letter)

    label[seq_len] = eos_decoder
    return label

def generate_seq(seq_len, num_repeat, num_tokens_rep, positive):
    """
    :param seq_len: length of the sequence
    :param num_repeat: number of times the token needs to be repeated
    :param repeat_dist: number of intervening tokens between reps
    :param num_tokens_rep: number of tokens that are repeated in the seq
    :return: sequence
    """
    # repeat position - recency; random but be balanced accross dataset

    seq_list = np.arange(0, max_seq_len)
    shuffle(seq_list)
    seq_list = seq_list[:seq_len]

    if(positive):
        # randomly generate first repeat position

        first_pos = randint(0, seq_len-1)
        # the repeated token is always at the end, there are no tokens after the
        # repeated token
        seq_list[first_pos] = seq_list[-1]
        rep_token = seq_list[first_pos]

    else:
        # none of the tokens are repeating
        rep_token = -1
        first_pos = -1


    return seq_list, rep_token, first_pos, seq_len

def aggregate_inputs(sequence, rep_token, first_token_pos, seq_len, positive):
    seq_list = tuple(sequence)
    if seq_list in seq_dict:
        skipped = 1
        return skipped
    else:
        skipped = 0
        seq_dict[seq_list] = 1

    # proceed to apend the sequence
    sequence_one_hot = []
    for token in sequence:
        seq_token = [0] * (max_seq_len + 1)
        seq_token[token] = 1
        sequence_one_hot.append(seq_token)
    sequence_one_hot.append(eos_seq_ip)
    x.append(sequence_one_hot)

    label = generate_labels(sequence)
    y.append(label)
    y_mlp.append(positive)

    token_repeated.append(rep_token)
    pos_first_token.append(first_token_pos)
    sequence_len.append(seq_len)

    return skipped


def generate_dataset(max_seq_len=26, num_tokens_rep=1):
    """
    :param num_samples:
    :param seq_len:
    :param num_repeat:
    :param repeat_dist:
    :param num_tokens_rep:
    :param max_seq_len:
    :return:
    """

    # min seq_len is always 2 as we do not consider 1 length sequence
    min_seq_len = 2
    num_repeat = 1

    num_positive_examples = 0
    num_negative_examples = 0

    for seq_len in range(min_seq_len, max_seq_len+1):
        #positive examples with repetion
        #print("seq_len is" + str(seq_len))
        num_samples = min((26*np.math.factorial(seq_len-1)), num_instances_per_seq_len)
        # number of samples per sequence
        samples_seq[seq_len] = num_samples*2
        for sample in range(num_samples):
            positive = 1
            sequence, rep_token, first_token_pos, seq_len = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)
            # while aggregating inputs do not add repeating samples
            if sequence is not None:
                skipped = aggregate_inputs(sequence, rep_token, first_token_pos, seq_len, positive)

            if(skipped == 0):
                num_positive_examples =num_positive_examples + 1
                #negative samples, only when we have added a positive sample
                positive = 0
                sequence, rep_token, first_token_pos, seq_len = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)

                skipped = aggregate_inputs(sequence, rep_token, first_token_pos, seq_len, positive)
                if(skipped == 0):
                    num_negative_examples = num_negative_examples + 1

    print("Number of positive examples are: " + str(num_positive_examples))
    print("Number of negative examples are: " + str(num_negative_examples))

    return x, y, y_mlp, token_repeated, pos_first_token, sequence_len


def decode_seq(x, y):
    batch = []
    num_samples = len(x)
    for sequence in range(num_samples):
        num_tokens = len(x[sequence])
        seq = []
        for token in range(num_tokens):
            seq.append(one_hot_decoding(alphabet, x[sequence][token]))
        batch.append(seq)

    seq_list = [(batch[seq], y[seq]) for seq in range(num_samples)]

    for sample in range(num_samples):
        print(" Sequence: " + str(seq_list[sample][0]) + "Target: " + str(
            seq_list[sample][1]))

def decode_seq_encoder_mlp(x_encoder, x_mlp, y):
    batch_encoder = []
    batch_mlp = []
    num_samples = len(x_encoder)
    for sequence_index in range(num_samples):
        num_tokens = len(x_encoder[sequence_index])
        seq = []
        for token in range(num_tokens):
            seq.append(one_hot_decoding(alphabet, x_encoder[sequence_index][token]))

        # decoder the query one hot encoded vector
        batch_mlp.append(one_hot_decoding(alphabet, x_mlp[sequence_index]))
        batch_encoder.append(seq)

    seq_list = [(batch_encoder[seq_index], batch_mlp[seq_index], y[seq_index])
                for seq_index in range(num_samples)]

# check some random samples
    for sample in range(10):
        seq_id = randint(0, num_samples)
        if batch_mlp[seq_id] in batch_encoder[seq_id]:
            if(y[seq_id] == 1):
                print("Test passed")
                print(" Sequence: " + str(seq_list[sample][0]) + " MLP input: " + str(batch_mlp[sample]) +
                        "Target: " + str(seq_list[sample][2]))
            else:
                print("Test Failed")
                print(" Sequence: " + str(seq_list[sample][0]) + " MLP input: " + str(batch_mlp[sample]) +
                        "Target: " + str(seq_list[sample][2]))
            print(sample)

def check_dataset(encoder_input_data_train, encoder_input_data_test,
                  mlp_input_data_train, mlp_input_data_test,
                  y_mlp_train, y_mlp_test,
                  sequence_len_train, sequence_len_test,
                  token_repeated_train, token_repeated_test,
                  pos_first_token_train, pos_first_token_test):

    # check the train data
    decode_seq_encoder_mlp(encoder_input_data_train, mlp_input_data_train, y_mlp_train)

# check if the train and the test dataset have any same sequences.
def check_train_test(encoder_input_data_train):

    num_rept_test = 0
    train_samples = []
    train_encoder = []
    num_samples = len(encoder_input_data_train)
    for sequence_index in range(num_samples):
        num_tokens = len(encoder_input_data_train[sequence_index])
        seq = []
        for token in range(num_tokens):
            seq.append(one_hot_decoding(alphabet, encoder_input_data_train[sequence_index][token]))
        train_encoder.append(seq)

    for seq in train_encoder:
        train_samples.append([''.join([i[0] for i in seq[0:-1]])])

    for seq in train_samples:
        count = train_samples.count(seq)
        if(count > 1):
            num_rept_test = num_rept_test+1

    return num_rept_test


def plot_data(x, y, token_repeated, pos_first_token):
    plt.figure()
    counts = np.bincount(np.array(token_repeated)[np.array(token_repeated)>=0])

    # Switching to the OO-interface. You can do all of this with "plt" as well.
    fig, ax = plt.subplots()
    plt.title("Histogram of the tokens repeated")
    plt.xlabel("Letter repeated")
    plt.ylabel("Number of samples in which token is repeated")
    ax.bar(range(max_seq_len), counts, width=1, align='center',edgecolor = 'black')
    ax.set(xticks=range(max_seq_len+1), xlim=[-1, max_seq_len])
    plt.savefig("Tokens histogram")
    plt.close()
    # Show plot
    #plt.show(block=True)


    # Switching to the OO-interface. You can do all of this with "plt" as well.
    start_index = 0
    end_index = 0

    for i in range(2, max_seq_len+1):
        # sequence length doesnt count the negative samples
        start_index = start_index + samples_seq[i-1]
        end_index = end_index + samples_seq[i]
        fig, ax = plt.subplots()
        plt.title("First position histogram: seq_len" + str(i))
        plt.xlabel("first position of repeated token")
        plt.ylabel("Number of samples")
        counts = np.bincount(np.array(pos_first_token[start_index:end_index:2]).astype('int64'))
        ax.bar(np.arange(i-1), counts, width=1, align='center', edgecolor='black')
        ax.set(xticks=np.arange(i-1), xlim=[0, max_seq_len])
        plt.savefig("First token position : seq_len" + str(i))
        plt.close()

"""
num_tokens_rep = 1
max_seq_len = 26

x, y, y_mlp, token_repeated, pos_first_token, seq_len = generate_dataset(max_seq_len,
                                                                      num_tokens_rep)


#decode_seq(x,y_mlp)
"""

# scatter plot of the seq len and position first token
#plt.plot(pos_first_token[0:len(pos_first_token):2], 'o')
#plt.plot(seq_len[0:len(seq_len):2], 'o')
#plt.show()

#plot_data(x, y, token_repeated, pos_first_token)
