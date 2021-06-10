from __future__ import unicode_literals, print_function, division
from RNN import *
from train import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_iters = 5000
num_samples = n_iters
seq_len = 5

# x - one hot vectors
# y - sequence of 0 and 1 based on whether the current letter is unrepeated
# or repeated
#x, y = generate_dataset(num_samples)

#train_pair = tensorsFromIpTarget(x, y)

# input is one-hot encoded vectors of the alphabet
input_size = 26
input_size_decoder = 1
hidden_size = 256
# output can either be 0: not repeated or 1: repeated 2: eos
output_size = 3
encoder1 = EncoderRNN(input_size, hidden_size, device).to(device)

decoder1 = DecoderRNN(input_size_decoder, hidden_size, output_size, device, dropout_p=0.1).to(device)

train_iters(encoder1, decoder1, seq_len, n_iters=n_iters, print_every=1, num_samples=num_samples)

def evaluate(encoder, decoder, seq, max_length = MAX_LENGTH):
    with torch.no_grad():
        input_tensor = seq
        input_length = input_tensor[0].size()[0]
        encoder_hidden = encoder.initHidden(device = device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                      device = device)


        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[0][i],
                                                     encoder_hidden)

            encoder_outputs[i] += encoder_output[0, 0]


        decoder_input = torch.tensor([[SOS_token_target]], device = device)

        decoder_hidden = encoder_hidden

        decoded_words = []

        for i in range(max_length+1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token_target:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach().float()

        return decoded_words




def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        x, y = generate_dataset(1, seq_len)

        #print('>', x)
        print('=', y)

        # (inp_one hot encoded, op_sequence)
        pair = [tensorsFromIpTarget(x[0], y[0])]


        output_words = evaluate(encoder, decoder, pair[0])
        #output_sentence = ' '.join(output_words)
        print('<', output_words)
        print('')

evaluateRandomly(encoder1, decoder1)