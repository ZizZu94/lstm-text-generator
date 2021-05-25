import data

##################### LOAD DATA FROM FILES #########################
data_folder_path = './data'
print('-- Loading data ..... --')
corpus = data.Corpus(data_folder_path)
print('Loaded training words: ', len(corpus.train))
print('Loaded validation words: ', len(corpus.valid))
print('Loaded testing words: ', len(corpus.test))
# without <eos> and <unk>
print('Vocabulary size: ', len(corpus.dictionary.idx2word) - 2)
print('-- Finished loading data --')
