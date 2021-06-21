"""
Generation of synthetic dataset to evaluate memory retention in neural models
"""

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
token_repeated = list()
pos_first_token = list()
repeat_dist = list()
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

    seq_list = np.arange(0, seq_len)
    shuffle(seq_list)

    if(positive):
        first_pos = randint(0, seq_len)
        #print("first position" + str(first_pos))

        if(first_pos == seq_len-1):
            # rare case when first pos is picked as the last pos in seq, then
            # force first pos to be 0.
            first_pos = 0

        if(first_pos+1 == seq_len-1):
            # rep pos for seq 2 is always fixed
            rep_pos = 1
        else:
            rep_pos = randint(first_pos+1, seq_len-1)

        rep_dist = rep_pos-first_pos
        first_token_pos = first_pos

        rep_token = seq_list[first_pos]
        seq_list[rep_pos] = seq_list[first_pos]

    else:
        # none of the tokens are repeating
        rep_token = -1
        first_token_pos = -1
        rep_dist = -1


    return seq_list, rep_token, first_token_pos, rep_dist

def aggregate_inputs(sequence, rep_token, first_token_pos, rep_dist):
    sequence_one_hot = []
    for token in sequence:
        seq_token = [0] * (max_seq_len + 1)
        seq_token[token] = 1
        sequence_one_hot.append(seq_token)
    sequence_one_hot.append(eos_seq_ip)
    x.append(sequence_one_hot)

    label = generate_labels(sequence)
    y.append(label)

    token_repeated.append(rep_token)
    pos_first_token.append(first_token_pos)
    repeat_dist.append(rep_dist)

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

    for seq_len in range(min_seq_len, max_seq_len):
        #positive examples with repetion
        #print("seq_len is" + str(seq_len))
        num_samples = min((26*np.math.factorial(seq_len-1)), 5000)
        for sample in range(num_samples):
            positive = 1
            sequence, rep_token, first_token_pos, rep_dist = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)

            aggregate_inputs(sequence, rep_token, first_token_pos, rep_dist)

            #negative samples
            positive = 0
            sequence, rep_token, first_token_pos, rep_dist = generate_seq(seq_len,
                                                                      num_repeat,
                                                                      num_tokens_rep,
                                                                       positive)

            aggregate_inputs(sequence, rep_token, first_token_pos, rep_dist)

    return x, y, token_repeated, pos_first_token, repeat_dist


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



num_tokens_rep = 1
max_seq_len = 26

x, y, token_repeated, pos_first_token, repeat_dist = generate_dataset(max_seq_len,
                                                                      num_tokens_rep)

decode_seq(x,y)

# Creating histogram
plt.hist(token_repeated, bins = max_seq_len)

# Show plot
plt.show(block=True)

# Creating histogram
plt.hist(pos_first_token, bins = max_seq_len)

# Show plot
plt.show(block=True)

# Creating histogram
plt.hist(repeat_dist, bins = max_seq_len)

# Show plot
plt.show(block=True)



"""
max_seq_len = 26
x_axis = range(0, max_seq_len)
token_list = []
seq_batch = []
seq_len=10
num_repeat=1
repeat_dist=5
num_tokens_rep=1
max_seq_len=26
for iter in range(10000):
    seq_list, rep_token = generate_sequence(seq_len, num_repeat, repeat_dist,
                                            num_tokens_rep, max_seq_len)
    #print(rep_token)
    token_list.append(rep_token)
    #print(seq_list)
    #print(token_list)

#print(token_list)


# Creating histogram
plt.hist(token_list, bins = max_seq_len)

# Show plot
plt.show(block=True)
"""


