import argparse

import torch
import numpy as np

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--save_path', type=str, default='/model/model.pt',
                    help='folder where we save pretained model weights')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
args = parser.parse_args()
args.cuda = True

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

data_folder_path = './data'
log_interval = 100

################### LOAD MODEL ###########################

save_path = './weights/lstm-model-weights.net'
with open(save_path, 'rb') as f:
    model = torch.load(f)


################### LOAD DATA and INPUT TEXT ###########################
print('=' * 90)
print('| Loading corpus ....')
print('=' * 90)
corpus = data.Corpus(data_folder_path)
unk_idx = corpus.dictionary.word2idx['<unk>']
print('=' * 90)
print('| Finished: corpus loaded ....')
print('=' * 90)
print('\n')

IO_folder = './text-generate'
input_words = []
input_text = ''
with open(IO_folder + '/input.txt', 'r') as file:
    for line in file:
        for word in line.split():
            input_words.append(word)

input_text = ' '.join(input_words)


################## GENERATE NEW TEXT ####################

def predict(net, input, h=None):
    output, h = net(input, h)
    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
    fav_two = torch.multinomial(word_weights, 2, replacement=False)

    # avoid <unk> token
    unk_id = corpus.dictionary.word2idx['<unk>']
    # with out <unk>
    word_idx = fav_two[0] if fav_two[0] != unk_id else fav_two[1]
    # remove comment below if want text with <unk> and comment previous line
    # word_idx = fav_two[0]

    return word_idx, h


def sample(net, size, prime='it is'):
    net.eval()

    # push to GPU
    if args.cuda:
        net.cuda()
    else:
        net.cpu()

    # batch size is 1
    h = net.init_hidden(1)

    words = prime.split()

    # predict next token of the input words
    for t in prime.split():
        if t in corpus.dictionary.word2idx:
            idx = np.array([[corpus.dictionary.word2idx[t]]])
        else:
            idx = np.array([[unk_idx]])

        with torch.no_grad():
            inp = torch.from_numpy(idx).long()
        if args.cuda:
            inp.data = inp.data.cuda()

        word_idx, h = predict(net, inp, h)

    words.append(corpus.dictionary.idx2word[word_idx])

    # input: next word of the input words
    with torch.no_grad():
        input = torch.tensor([[word_idx]]).long()
    if args.cuda:
        input.data = input.data.cuda()

    # predict subsequent tokens and write in the output file
    with open(IO_folder + '/output.txt', 'w') as outf:
        # write input words in the file
        for i, word in enumerate(words):
            word = '\n' if word == "<eos>" else word
            #outf.write(word + ('\n' if i % 20 == 19 else ' '))
            outf.write(word + ('' if word == '\n' else ' '))
        for i in range(size-len(words)-1):
            word_idx, h = predict(net, input, h)

            word = corpus.dictionary.idx2word[word_idx]
            word = '\n' if word == "<eos>" else word

            #outf.write(word + ('\n' if i % 20 == 19 else ' '))
            outf.write(word + ('' if word == '\n' else ' '))

            if i % log_interval == 0:
                print('| Generated {}/{} words'.format(i, size))

            input.data.fill_(word_idx)
            words.append(word)

    return ' '.join(words)


new_text = sample(model, args.words, input_text)
print('\n')
print('=' * 90)
print('| Text generated: {} words'.format(args.words))
print('=' * 90)
print(new_text)
