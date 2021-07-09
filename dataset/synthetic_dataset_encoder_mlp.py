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
sequence_length = [0] * 27
max_seq_len = 26

eos_seq_ip = [0] * (max_seq_len + 1)
eos_seq_ip[-1] = 1

eos_decoder = 2

alphabet = 'abcdefghijklmnopqrstuvwxyz'
num_to_letter = {}
num_to_letter[26] = "eos"
for num, letter in enumerate(alphabet):
    num_to_letter[num] = letter

def one_hot_decoding(alphabet, one_hot_ip):

    seq = list()
    pos = 0
    for val in one_hot_ip:
        if val == 1:
            letter = num_to_letter[pos]
            break
        pos = pos+1

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

    seq_list = np.arange(0, 26)
    shuffle(seq_list)
    seq_list = seq_list[:seq_len]

    if(positive):
        # randomly generate first repeat position and the repeat position
        #first_pos, rep_pos = randint(0, seq_len, 2)
        #first_pos, rep_pos = random.sample(range(0, seq_len), 2)

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


    min_seq_len = 2
    num_repeat = 1

    for seq_len in range(min_seq_len, max_seq_len+1):
        #positive examples with repetion
        #print("seq_len is" + str(seq_len))
        num_samples = min((26*np.math.factorial(seq_len-1)), 5000)
        sequence_length[seq_len] = num_samples*2
        for sample in range(num_samples):
            positive = 1
            sequence, rep_token, first_token_pos, seq_len = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)

            if sequence is not None:
                aggregate_inputs(sequence, rep_token, first_token_pos, seq_len, positive)

            #negative samples, no repetition
            positive = 0
            sequence, rep_token, first_token_pos, seq_len = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)

            aggregate_inputs(sequence, rep_token, first_token_pos, seq_len, positive)

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


def plot_data(x, y, token_repeated, pos_first_token):
    plt.figure()
    counts = np.bincount(np.array(token_repeated)[np.array(token_repeated)>=0])

    # Switching to the OO-interface. You can do all of this with "plt" as well.
    fig, ax = plt.subplots()
    plt.title("Histogram of the tokens repeated")
    plt.xlabel("Letter repeated")
    plt.ylabel("Number of samples in which token is repeated")
    ax.bar(range(26), counts, width=1, align='center',edgecolor = 'black')
    ax.set(xticks=range(27), xlim=[-1, 26])
    plt.savefig("Tokens histogram")
    plt.close()
    # Show plot
    #plt.show(block=True)


    # Switching to the OO-interface. You can do all of this with "plt" as well.
    start_index = 0
    end_index = 0

    for i in range(2, 27):
        # sequence length doesnt count the negative samples
        start_index = start_index + sequence_length[i-1]
        end_index = end_index + sequence_length[i]
        fig, ax = plt.subplots()
        plt.title("First position histogram: seq_len" + str(i))
        plt.xlabel("first position of repeated token")
        plt.ylabel("Number of samples")
        counts = np.bincount(np.array(pos_first_token[start_index:end_index:2]).astype('int64'))
        ax.bar(np.arange(i-1), counts, width=1, align='center', edgecolor='black')
        ax.set(xticks=np.arange(i-1), xlim=[0, 26])
        plt.savefig("First token position : seq_len" + str(i))
        plt.close()

"""
num_tokens_rep = 1
max_seq_len = 26

x, y, y_mlp, token_repeated, pos_first_token, seq_len = generate_dataset(max_seq_len,
                                                                      num_tokens_rep)


decode_seq(x,y)

# scatter plot of the seq len and position first token
#plt.plot(pos_first_token[0:len(pos_first_token):2], 'o')
#plt.plot(seq_len[0:len(seq_len):2], 'o')
#plt.show()

plot_data(x, y, token_repeated, pos_first_token)

"""