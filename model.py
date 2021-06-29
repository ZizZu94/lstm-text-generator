import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, vocab_num, emb_dim, hidden_size, num_layers=2, dropout_prob=0.2):
        super(LSTMModel, self).__init__()

        self.vocab_num = vocab_num  # num output
        self.emb_dim = emb_dim  # num input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # embedding layer
        self.emb_layer = nn.Embedding(
            self.vocab_num,
            self.emb_dim
        )
        # lstm layer
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
        )

        # dropout
        self.dropout = nn.Dropout(
            dropout_prob
        )
        # fully connected layer
        self.fc_layer = nn.Linear(
            self.hidden_size,
            self.vocab_num
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.emb_layer.weight.data.uniform_(-initrange, initrange)
        self.fc_layer.bias.data.fill_(0)
        self.fc_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.emb_layer(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)

        output_reshaped = output.view(
            output.size(0)*output.size(1), output.size(2))

        fc_out = self.fc_layer(output_reshaped)

        return fc_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_size).cuda(), weight.new_zeros(self.num_layers, batch_size, self.hidden_size).cuda()
