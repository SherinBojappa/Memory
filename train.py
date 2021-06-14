"""

Training for the seq to seq model

"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

#import sys
#sys.path.apend("./dataset")
#import dataset.dataset_generation.py as data
from dataset.memory_dataset_generation import *
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 11

EOS_token_input = [1.0]*26
EOS_token_target = 2.0
SOS_token_target = -1.0


# append EOS at the end of the sequence and convert it into a tensor
def tensorFromSequence(seq, ip_type):
    if ip_type == 0:
        seq.append(EOS_token_input)
    elif ip_type == 1:
        seq.append(EOS_token_target)
    return torch.tensor(seq, dtype = torch.float, device = device)

def tensorsFromIpTarget(input, target):
    input_tensor = tensorFromSequence(input, 0)
    target_tensor = tensorFromSequence(target, 1)

    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device, max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # input_tensor = input_tensor.view(-1,1)
    # target_tensor = target_tensor.view(-1, 1)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)


    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0][0]
        #print("Encoder op size is: ")
        #print(encoder_output.size())
        #print("Input length is: ")
        #print(input_length)



    decoder_input = torch.tensor([[SOS_token_target]], device = device)

    decoder_hidden = encoder_hidden

    loss = 0

    for i in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input,
                                decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach().float()

        #print("here")
        #print(decoder_output.size())
        #print(target_tensor[i].size())
        loss += criterion(decoder_output, target_tensor[i].view(-1).long())

        if decoder_input.item() == EOS_token_target:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base = 0.2)
    #ax.y_axis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def train_iters(encoder, decoder, seq_len, n_iters, print_every = 1000, plot_every=1000,
                learning_rate = 0.001, num_samples=100, num_repeat=1, repeat_dist=1,
                            num_tokens_rep=1, max_seq_len=26):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    # x - one hot vectors
    # y - sequence of 0 and 1 based on whether the current letter is unrepeated
    # or repeated

    x, y = generate_dataset(num_samples, seq_len, num_repeat, repeat_dist,
                            num_tokens_rep, max_seq_len)


    #train_pair = tensorsFromIpTarget(x, y)

    #training_pairs =[]

    #for i in range(n_iters):
    #    training_pairs.append(tensorsFromIpTarget(x[i], y[i]))

    # (inp_one hot encoded, op_sequence)
    training_pairs = [tensorsFromIpTarget(x[i], y[i]) for i in range(num_samples)]
    print("size of training pairs: ")
    print(training_pairs[0][0].size())

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion, device)

        print_loss_total += loss
        plot_loss_total += loss


        if iter % print_every == 0:
            # average loss for print_every iteration
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0

            print('(%d %d%%) %.4f' %(iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)











