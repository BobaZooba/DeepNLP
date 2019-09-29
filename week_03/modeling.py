import torch
from torch import nn


class RNNCell(nn.Module):

    def __init__(self, in_features, hidden_features):

        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features

        self.input_linear = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.memory_linear = nn.Linear(in_features=hidden_features, out_features=hidden_features)

    def init_memory(self, batch_size):

        return torch.zeros((batch_size, self.hidden_features))

    def forward(self, x, memory=None):

        if memory is None:
            memory = self.init_memory(batch_size=x.shape[0])

        x = torch.tanh(self.input_linear(x) + self.memory_linear(memory))

        memory = x

        return x, memory


class LSTMCell(nn.Module):

    def __init__(self, in_features, hidden_features):

        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features

        self.forget_gate_input = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.forget_gate_hidden = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features)

        self.in_gate_input = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.in_gate_hidden = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features)

        self.cell_gate_input = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.cell_gate_hidden = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features)

        self.out_gate_input = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.out_gate_hidden = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features)

    def init_memory(self, batch_size):

        hidden, cell = torch.zeros((batch_size, self.hidden_features)), torch.zeros((batch_size, self.hidden_features))

        return hidden, cell

    def forward(self, x, memory=None):

        if memory is None:
            hidden, cell = self.init_memory(batch_size=x.shape[0])
        else:
            hidden, cell = memory

        forget_gate = torch.sigmoid(self.forget_gate_input(x) + self.forget_gate_hidden(hidden))
        cell = forget_gate * cell

        in_gate = torch.sigmoid(self.in_gate_input(x) + self.in_gate_hidden(hidden))
        cell_gate = torch.tanh(self.cell_gate_input(x) + self.cell_gate_hidden(hidden))

        new_cell = in_gate * cell_gate

        cell = cell + new_cell

        output_cell = torch.tanh(cell)
        output_gate = torch.sigmoid(self.out_gate_input(x) + self.out_gate_hidden(hidden))

        hidden = output_cell * output_gate

        return hidden, (hidden, cell)


class RNN(nn.Module):

    def __init__(self, rnn_cell, output_last=False):

        super().__init__()

        self.rnn_cell = rnn_cell

        self.output_last = output_last

    def forward(self, x, memory=None):

        if memory is None:
            memory = self.rnn_cell.init_memory(batch_size=x.shape[0])

        hiddens = []

        for timestamp in range(x.size(1)):
            current_timestamp = x[:, timestamp, :]
            current_hidden, memory = self.rnn_cell(current_timestamp, memory)
            hiddens.append(current_hidden.unsqueeze(1))

        if self.output_last:
            return hiddens[-1].squeeze()

        hiddens = torch.cat(hiddens, dim=1)

        return hiddens


class BidirectionalRNN(nn.Module):

    def __init__(self, rnn_cell_forward, rnn_cell_backward, output_last=False):

        super().__init__()

        self.rnn_cell_forward = rnn_cell_forward
        self.rnn_cell_backward = rnn_cell_backward

        self.output_last = output_last

    def forward(self, x, forward_memory=None, backward_memory=None):

        if forward_memory is None:
            forward_memory = self.rnn_cell_forward.init_memory(batch_size=x.shape[0])

        if backward_memory is None:
            backward_memory = self.rnn_cell_backward.init_memory(batch_size=x.shape[0])

        sequence_length = x.size(1)

        forward_hiddens = []
        backward_hiddens = []

        for timestamp in range(sequence_length):

            forward_timestamp = x[:, timestamp, :]
            backward_timestamp = x[:, sequence_length - timestamp - 1, :]

            forward_hidden, forward_memory = self.rnn_cell_forward(forward_timestamp, forward_memory)
            backward_hidden, backward_memory = self.rnn_cell_backward(backward_timestamp, backward_memory)

            forward_hiddens.append(forward_hidden.unsqueeze(1))
            backward_hiddens.append(backward_hidden.unsqueeze(1))

        if self.output_last:
            return torch.cat([forward_hiddens[-1], backward_hiddens[-1]], dim=1)

        forward_hiddens = torch.cat(forward_hiddens, dim=1).unsqueeze(1)
        backward_hiddens = torch.cat(backward_hiddens, dim=1).unsqueeze(1)

        hiddens = torch.cat([forward_hiddens, backward_hiddens], dim=1)

        return hiddens
