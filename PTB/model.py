import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.dropout_rate,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, z_dim, pad_idx):
        super(LSTMEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size * 2, z_dim * 2)

    def forward(self, input_seq, length):
        # embed input
        embedded_input = self.embedding(input_seq)

        # RNN forward
        pack_input = pack_padded_sequence(embedded_input, length,
                                          batch_first=True)
        _, (h, c) = self.rnn(pack_input)

        # produce mu and logvar
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.output(hidden), 2, dim=-1)

        return mu, logvar


class RNNVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNNVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder = LSTMEncoder(vocab_size, embed_size,
                                   hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.init_h = nn.Linear(z_dim, hidden_size)
        self.init_c = nn.Linear(z_dim, hidden_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]

        # encode
        mu, logvar = self.encoder(enc_input, sorted_len)
        z = self.reparameterize(mu, logvar)

        # decode
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        pack_output, _ = self.rnn(pack_input, hidden)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu, logvar


class SelfVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(SelfVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.enc_embedding = nn.Embedding(vocab_size, embed_size,
                                          padding_idx=self.pad_idx)
        self.enc_rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.latent = nn.Linear(hidden_size * 2, z_dim * 2)

        # decoder
        self.dec_embedding = nn.Embedding(vocab_size, embed_size,
                                          padding_idx=self.pad_idx)
        self.attn = nn.Linear(hidden_size + embed_size, self.time_step)
        self.combine = nn.Linear(hidden_size + embed_size + z_dim * 2,
                                 hidden_size)
        self.dec_rnn = nn.LSTMCell(hidden_size, hidden_size)

        # variational inference
        self.pri = nn.Linear(hidden_size, z_dim * 2)
        self.inf = nn.Linear(hidden_size * 2, z_dim * 2)
        self.aux = nn.Linear(z_dim, hidden_size)

        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def gaussian_kld(self, mu_left, logvar_left, mu_right, logvar_right):
        """
        compute KL(N(mu_left, logvar_left) || N(mu_right, logvar_right))
        """
        gauss_klds = 0.5 * (logvar_right - logvar_left +
                            logvar_left.exp() / logvar_right.exp() +
                            (mu_left - mu_right).pow(2) / logvar_right.exp() -
                            1.)
        return torch.sum(gauss_klds, 1)

    def forward(self, enc_input, dec_input, length):
        # process input
        batch_size = enc_input.size(0)
        max_len = torch.max(length)
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]
        att_mask = enc_input==self.pad_idx

        # encode
        enc_embedded = self.enc_embedding(enc_input)
        enc_input = pack_padded_sequence(enc_embedded, sorted_len,
                                         batch_first=True)
        pack_output, (h, c) = self.enc_rnn(enc_input)
        enc_output, _ = pad_packed_sequence(pack_output, batch_first=True)
        if max_len != self.time_step:
            padding = torch.zeros([batch_size,
                                   self.time_step - max_len,
                                   self.hidden_size], device=enc_output.device)
            enc_output = torch.cat([enc_output, padding], dim=1)
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.latent(hidden), 2, dim=-1)
        z = self.reparameterize(mu, logvar)

        # decode
        dec_embedded = self.dec_embedding(dec_input)
        dec_input = F.dropout(dec_embedded, p=self.dropout_rate,
                              training=self.training)
        hx = torch.zeros(batch_size, self.hidden_size, device=z.device)
        hidden = (hx, hx)

        outputs, klds, aux_cs = [], [], []
        for t in range(max_len + 1):
            x_step = dec_input[:, t]
            mask = (length + 1 > t).float()

            # prior
            pri_mu, pri_logvar = torch.chunk(self.pri(hidden[0]), 2, dim=-1)

            # attention mechanism
            scale = self.attn(torch.cat([x_step, hidden[0]], dim=-1))
            scale = scale.masked_fill(att_mask, -math.inf)
            attn_weight = F.softmax(scale, dim=-1)
            context = torch.bmm(attn_weight.unsqueeze(1),
                                enc_output).squeeze(1)

            # inference
            inf_mu, inf_logvar = torch.chunk(
                self.inf(torch.cat([hidden[0], context], dim=-1)), 2, dim=-1)
            klds.append(self.gaussian_kld(inf_mu, inf_logvar,
                                          pri_mu, pri_logvar) * mask)
            if self.training:
                z_step = self.reparameterize(inf_mu, inf_logvar)
            else:
                z_step = pri_mu

            # auxiliary
            aux_mu = self.aux(z_step)
            aux_cs.append(
                torch.sum((context.detach() - aux_mu).pow(2), 1))

            # RNN forward
            input = self.combine(
                torch.cat([x_step, aux_mu, z_step, z], dim=-1))
            hidden = self.dec_rnn(input, hidden)
            outputs.append(hidden[0])
        output = torch.stack(outputs, dim=1)
        kld = torch.stack(klds, dim=1).sum()
        aux_loss = torch.stack(aux_cs, dim=1).sum()
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx, :max_len + 1]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu, logvar, kld, aux_loss
