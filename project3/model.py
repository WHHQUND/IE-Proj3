import torch
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class LSTM_ASR_Discrete(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size=257, alphabet_size=28):

        super(LSTM_ASR_Discrete, self).__init__()
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(embedding_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)#, num_layers=2, dropout=0.4)

        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):
        
        embeds = self.word_embeddings(x)

        out = self.relu(self.lin1(embeds))

        out = torch.nn.utils.rnn.pack_padded_sequence(out, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(out)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)

        return log_probs

class LSTM_ASR_MFCC(torch.nn.Module):

    def __init__(self, mfcc_dim, hidden_dim, alphabet_size=28):

        super(LSTM_ASR_MFCC, self).__init__()
        self.mfcc_dim = mfcc_dim

        self.conv1 = nn.Conv1d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv1d(10, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(mfcc_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)#, num_layers=2, dropout=0.4)
        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):

        batch_size = x.size()[0]
        seq_length = x.size()[1]
        
        x = x.view(batch_size*seq_length, 1, self.mfcc_dim)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.view(batch_size, seq_length, self.mfcc_dim)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)

        return log_probs
