import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model

parser = argparse.ArgumentParser(
    description='Language generation with RNN-LSTM')
parser.add_argument('--embsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

save_path = './weights/lstm-model-weights.net'
args.cuda = True

##################### LOAD DATA FROM FILES #########################

data_folder_path = './data'

print('=' * 90)
print('| Loading data .....')
print('=' * 90)

corpus = data.Corpus(data_folder_path)
print('| Loaded training words: ', len(corpus.train))
print('| Loaded validation words: ', len(corpus.valid))
print('| Loaded testing words: ', len(corpus.test))
# vocab size without <eos> and <unk>
print('| Vocabulary size: ', len(corpus.dictionary) - 2)

print('=' * 90)
print('| Finished loading data')
print('=' * 90)

eval_batch_size = 10
train_data = data.make_batch(corpus.train, args.batch_size, args.cuda)
val_data = data.make_batch(corpus.valid, eval_batch_size, args.cuda)
test_data = data.make_batch(corpus.test, eval_batch_size, args.cuda)


##################### CREATE MODEL #########################

vocab_num = len(corpus.dictionary)
model = model.LSTMModel(vocab_num, args.embsize,
                        args.nhid, args.nlayers, args.dropout)

if args.cuda:
    model.cuda()
else:
    model.cpu()

print('=' * 90)
print('| Model structure')
print('=' * 90)
print(model)
print('=' * 90)
print('\n')

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

##################### TRAINING #########################

interval = 200  # interval to report
clip_grad_norm = 0.25


def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    return data, target


def evaluate(data_source):
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        # hidden size(nlayers, bsz, hdsize)
        hidden = model.init_hidden(eval_batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):  # iterate over every timestep
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


def train():
    # choose a optimizer

    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.data

        # log current situation
        if batch % interval == 0 and batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, lr,
                      elapsed * 1000 / interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


print('=' * 90)
print('| Training started')
print('=' * 90)
# Loop over epochs.
lr = args.lr
best_val_loss = None
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr)

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 90)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            for group in optimizer.param_groups:
                group['lr'] = lr

except KeyboardInterrupt:
    print('-' * 90)
    print('Exiting from training early')

# Load the best saved model.
with open(save_path, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 90)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 90)
