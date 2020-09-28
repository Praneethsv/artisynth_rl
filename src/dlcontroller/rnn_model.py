import torch
import torch.nn as nn

from torch.distributions import Normal

epsilon = 1e-6


class NormalVAE(nn.Module):
    def __init__(self, rnn_type, hidden_size, latent_size, input_sequence_length, output_sequence_length,
                 num_layers=1, action_space=None, bidirectional=False):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.latent_size = latent_size
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.bidirectional = bidirectional

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError

        self.encoder_rnn = rnn(self.input_sequence_length, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if self.bidirectional else 1) * num_layers

        self.decoder_rnn = rnn(self.input_sequence_length, hidden_size,  num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.direct_decoder_rnn = rnn(self.latent_size, self.output_sequence_length, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, hidden_size * self.hidden_factor)
        self.latent2logprob = nn.Linear(self.latent_size, hidden_size * self.hidden_factor)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        _, hidden = self.encoder_rnn(input_sequence)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        # z = to_var(torch.randn([batch_size, self.latent_size]))
        # z = z * std + mean
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        z = Normal(mean, std)
        z_t = z.rsample()
        z_log_prob = z.log_prob(z_t)

        # z = z.view(1, 1, -1)
        # DECODER
        hidden = self.latent2hidden(z_t)
        hidden_log_prob = self.latent2logprob(z_log_prob)
        print('latent2hidden shape: ', hidden.shape)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
            hidden = hidden.view(1, 1, -1)
        print('decoder hidden shape: ', hidden.shape)
        # decoder forward pass
        # outputs, _ = self.direct_decoder_rnn(z, self.output_sequence_length)
        outputs, _ = self.decoder_rnn(input_sequence, hidden)
        print('outputs size is: ', outputs.shape)
        y_t = torch.tanh(outputs)  # muscle activations
        actions = y_t * self.action_scale + self.action_bias
        return actions, mean, std, z, y_t, hidden_log_prob

    def sample(self, input_sequence):
        print('input sequence shape: ', input_sequence.shape)
        input_sequence = input_sequence.view(1, 1, -1)  # here this step is not sure but it removed the error "RuntimeError: input must have 3 dimensions, got 2"
        print('input sequence shape after view: ', input_sequence.shape)
        action_sequence, mean, std, z, y_t, log_prob = self.forward(input_sequence)
        log_prob = log_prob.unsqueeze(0)
        print('log_prob shape is: ', log_prob.shape)
        y_t = y_t.squeeze(0)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action_sequence, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(NormalVAE, self).to(device)
