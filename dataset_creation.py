# creates synthetic dataset used to evaluate the memory retention task

from dataset.synthetic_dataset_encoder_mlp import *
import csv

# only one repeated token
num_tokens_rep = 1
# need to also change this value in synthetic_dataset_encoder_mlp.py
max_seq_len = 100
num_instances_per_seq_len = 5000

x, y, y_mlp, raw_sequence, token_repeated, pos_first_token, sequence_len = generate_dataset(max_seq_len,
                                                                      num_tokens_rep, num_instances_per_seq_len)

# for the encoder mlp problem we split the last token as the query token
sequence_len = [length_seq -1 for length_seq in sequence_len]

# for x the last sample consists of what is fed into the mlp y
with open('memory_retention_raw.csv', 'w') as csvfile:
    fieldnames = ['index', 'seq_len', 'seq', 'token_repeated', 'rep_token_first_pos', 'query_token', 'target_val' ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    total_samples = len(x)
    for sample in range(total_samples):
        writer.writerow({'index': sample, 'seq_len': sequence_len[sample], 'seq':raw_sequence[sample][:-1],
                         'token_repeated':token_repeated[sample],
                         'rep_token_first_pos': pos_first_token[sample], 'query_token':raw_sequence[sample][-1],
                         'target_val':y_mlp[sample]})

csvfile.close()

