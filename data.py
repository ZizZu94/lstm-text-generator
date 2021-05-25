import os
import torch


class Dictionary(object):
    """
        Build word --> id and id --> word mapper
    """

    def __init__(self):
        self.word2idx = {}  # <dict> word: index
        self.idx2word = []  # array list[index] = word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
        Tokenizer
    """

    def __init__(self, data_path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(data_path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(data_path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(data_path, 'test.txt'))

    def tokenize(self, path):
        """
            Tokenize a text file
            input: path --> path of the file
        """
        assert os.path.exists(path)
        # Add all words to the dictionary
        with open(path, 'r') as file:
            token_counter = 0
            for line in file:
                for word in line.split() + ['<eos>']:
                    self.dictionary.add_word(word)
                    token_counter += 1

        # Tokenize the file
        with open(path, 'r') as file:
            ids = torch.LongTensor(token_counter)
            token_index = 0
            for line in file:
                for word in line.split() + ['<eos>']:
                    ids[token_index] = self.dictionary.word2idx[word]
                    token_index += 1

        # return tokenized ids
        return ids
