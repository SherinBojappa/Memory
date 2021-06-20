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

alphabet = 'abcdefghijklmnopqrstuvwxyz'
num_to_letter = {}
num_to_letter[0] = "eos"
for num, letter in enumerate(alphabet):
    num_to_letter[num + 1] = letter

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
    label = [0]*seq_len
    for num, letter in enumerate(sequence):
        #print(num, letter)
        if letter in new_list:
            label[num] = 1

        new_list.append(letter)

    return label

def generate_sequence(seq_len, num_repeat=1, repeat_dist=1,
                      num_tokens_rep=1, max_seq_len=26):
    """
    :param seq_len: length of the sequence
    :param num_repeat: number of times the token needs to be repeated
    :param repeat_dist: number of intervening tokens between reps
    :param num_tokens_rep: number of tokens that are repeated in the seq
    :return: sequence
    """
    # repeat position - recency; random but be balanced accross dataset

    seq_list = np.arange(1, max_seq_len+1)
    shuffle(seq_list)
    seq_list = seq_list[:seq_len]


    #TODO may need to adjust the program to  repeat if num_repeat >1
    # insert the token to repeat
    # fix 1st repeat position
    rep_pos = randint(0, seq_len-1)
    #print(rep_pos)
    # TODO multiple repeats

    while(1):
        if(rep_pos + repeat_dist+1 >= seq_len):
            if( rep_pos - repeat_dist-1 < 0):
                rep_pos = randint(0, seq_len - 1)
                print("In while loop")
                print(rep_pos)
            else:
                break
        else:
            break

    #print("rep pos " + str(rep_pos))
    rep_token = seq_list[rep_pos]
    #print(rep_token)

    seq_list[rep_pos] = rep_token
    for i in range(num_repeat):
        if rep_pos+((i+1)*(repeat_dist+1)) < seq_len:
            seq_list[rep_pos+((i+1)*repeat_dist)+1] = rep_token
            #print('plus')
            #print(rep_pos+((i+1)*(repeat_dist+1)))
        elif rep_pos-((i+1)*(repeat_dist))-1 < seq_len:
            #print('minus')
            #print(rep_pos-((i+1)*(repeat_dist))-1)
            seq_list[rep_pos-((i+1)*(repeat_dist))-1] = rep_token

    #for letter in seq_list:
        #print(alphabet[letter])

    return seq_list, rep_token

def generate_dataset(num_samples, seq_len, num_repeat, repeat_dist,
                     num_tokens_rep, max_seq_len):
    x = list()
    y = list()

    for iter, sample in enumerate(range(num_samples)):
        sequence, rep_token = generate_sequence(seq_len, num_repeat, repeat_dist,
                                     num_tokens_rep, max_seq_len)
        #print(sequence)
        #for letter in sequence:
            #print(alphabet[letter])

        sequence_one_hot = []
        for token in sequence:
            seq_token = [0]*(max_seq_len+1)
            seq_token[token] = 1
            sequence_one_hot.append(seq_token)
        x.append(sequence_one_hot)

        label = generate_labels(sequence)
        y.append(label)

    return x, y

def decode_seq(x, y, num_samples, seq_len):
    batch = []
    for sequence in range(num_samples):

        seq = []
        for token in range(seq_len):
            seq.append(one_hot_decoding(alphabet, x[sequence][token]))
        batch.append(seq)

    seq_list = [(batch[seq], y[seq]) for seq in range(num_samples)]

    for sample in range(num_samples):
        print(" Sequence: " + str(seq_list[sample][0]) + "Target: " + str(
            seq_list[sample][1]))

"""
num_samples = 1000
seq_len = 10
num_repeat = 1
repeat_dist = 2
num_tokens_rep = 1
max_seq_len = 26

x, y = generate_dataset(num_samples, seq_len, num_repeat, repeat_dist,
                     num_tokens_rep, max_seq_len)

decode_seq(x,y, num_samples, seq_len)

"""

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


