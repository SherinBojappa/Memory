"""
Toy dataset to validate the memory retention model.
This consists of input sequences generated from the alphabet A-Z and output
sequences are 0 or 1 depending on whether the character is seen before or not
"""

from string import ascii_lowercase
from random import choice

#seq_len = 10
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def generate_sequence(seq_len):
    return [choice(ascii_lowercase) for _ in range(seq_len)]


def letter_to_num_map(alphabet):
    letter_to_num = {}
    for num, letter in enumerate(alphabet):
        letter_to_num[letter] = num

    return letter_to_num


def one_hot_decoding(alphabet, one_hot_ip):
    num_to_letter = {}
    for num, letter in enumerate(alphabet):
        num_to_letter[num] = letter

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


def generate_dataset(num_samples, seq_len):
    letter_to_num = letter_to_num_map(alphabet)
    x = list()
    y = list()
    for iter, sample in enumerate(range(num_samples)):
        sequence = generate_sequence(seq_len)
        #print(sequence)
        sequence_one_hot = one_hot_encoding(sequence, letter_to_num)
        # print(sequence_one_hot)
        label = generate_labels(sequence)
        # print(label)
        x.append(sequence_one_hot)
        y.append(label)

    return x, y

"""
num_samples = 10
seq_len = 20

x, y = generate_dataset(num_samples, seq_len)

batch = []
for sequence in range(num_samples):

    seq = []
    for token in range(seq_len):
       seq.append(one_hot_decoding(alphabet, x[sequence][token]))
    batch.append(seq)



seq_list = [(batch[seq], y[seq]) for seq in range(num_samples)]

for sample in range(num_samples):
    print(" Sequence: " + str(seq_list[sample][0]) + "Target: " + str(seq_list[sample][1]))
"""