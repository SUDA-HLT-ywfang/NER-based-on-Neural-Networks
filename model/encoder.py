import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class Encoder(nn.Module):
    def __init__(self, feature_extractor, embedding_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.feature_extractor_name = feature_extractor
        if feature_extractor == 'BiLSTM':
            self.feature_extractor = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim//2,
                bidirectional=True,
                batch_first=True,
                num_layers=num_layers
            )
        elif feature_extractor == 'LSTM':
            self.feature_extractor = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                bidirectional=False,
                batch_first=True,
                num_layers=num_layers
            )
        elif feature_extractor == 'CNN':
            self.feature_extractor = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=hidden_dim,
                kernel_size=3
            )
        else:
            raise RuntimeError("Unrecognized feature extractor name...")

    def forward(self, word_embeddings, sorted_sent_length):
        if self.feature_extractor_name == 'BiLSTM' or self.feature_extractor_name == 'LSTM':
            rnn_input = pack_padded_sequence(word_embeddings, sorted_sent_length, batch_first=True)
            rnn_out,h_n = self.feature_extractor(rnn_input, None)
            rnn_out,_ = pad_packed_sequence(rnn_out, batch_first=True)
            return rnn_out
        elif self.feature_extractor_name == 'CNN':
            batch_size = word_embeddings.size(0)
            # input:(N, in_channels, length_of_sequence)
            cnn_out = self.feature_extractor(word_embeddings.transpose(2, 1))
            # output:(N, out_channels, length_of_sequence)
            cnn_out_pooling = nn.functional.max_pool1d(cnn_out, cnn_out.size(2)).view(batch_size, -1)
            return cnn_out_pooling

    def get_last_hidden(self, word_embeddings, sorted_sent_length):
        if self.feature_extractor_name == 'BiLSTM' or self.feature_extractor == 'LSTM':
            rnn_input = pack_padded_sequence(word_embeddings, sorted_sent_length, batch_first=True)
            rnn_out, h_c_n = self.feature_extractor(rnn_input, None)
            last_hidden_state = h_c_n[0].transpose(1, 0)
            batch_size = word_embeddings.size(0)
            last_hidden_state_fb = torch.reshape(last_hidden_state, (batch_size, -1))
            return last_hidden_state_fb
        elif self.feature_extractor_name == 'CNN':
            batch_size = word_embeddings.size(0)
            # input:(N, in_channels, length_of_sequence)
            cnn_out = self.feature_extractor(word_embeddings.transpose(2, 1))
            # output:(N, out_channels, length_of_sequence)
            cnn_out_pooling = nn.functional.max_pool1d(cnn_out, cnn_out.size(2)).view(batch_size, -1)
            return cnn_out_pooling
