"""ICONIP — hard-attention models (RAM baseline, DRAM baseline, MRAM main)."""
import torch
import torch.nn as nn

import modules


class PeripheralCNN(nn.Module):
    """Small CNN used by DRAM to initialise the upper recurrent state with a
    global-context feature (Ba 2015, "Multiple Object Recognition with Visual
    Attention"). For 8×8 patches; output dim matches the LSTM hidden size."""

    def __init__(self, in_channels=3, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RecurrentAttention(nn.Module):
    """RAM — Recurrent Model of Visual Attention (Mnih et al. 2014).

    A single LSTM integrates sequential glimpses; the same hidden state
    feeds the classifier, location policy, and baseline.
    """

    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        super().__init__()
        self.std = std
        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, h_h, c_t_prev, c_h_prev, sigma=None):
        g_t = self.sensor(x, l_t_prev)
        out, (h_t, c_t) = self.rnn(g_t.unsqueeze(0), (h_t_prev, c_t_prev))
        out = out.squeeze(0)
        log_pi, l_t, entropy_mean = self.locator(out)
        b_t = self.baseliner(out).squeeze()
        log_probas = self.classifier(out)
        return h_t, h_h, c_t, c_h_prev, l_t, b_t, log_probas, log_pi, None, None, None, None, entropy_mean


class DRAM(nn.Module):
    """DRAM — Deep Recurrent Attention Model (Ba 2015).

    Two stacked LSTMs; the upper recurrent state is seeded with a CNN-derived
    global-context feature on the first step. Classification reads from the
    LOWER state (out1), location from the UPPER state (out2).
    """

    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        super().__init__()
        self.std = std
        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.periphery = PeripheralCNN(in_channels=c, out_dim=hidden_size)
        self.R1 = nn.LSTM(hidden_size, hidden_size)
        self.R2 = nn.LSTM(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, h_h_prev, c_t_prev, c_h_prev, sigma=None):
        g_t = self.sensor(x, l_t_prev)
        if h_h_prev.sum().detach().cpu().numpy() == 0:
            h_h_prev = self.periphery(x).unsqueeze(0)
        out1, (h_t, c1) = self.R1(g_t.unsqueeze(0), (h_t_prev, c_t_prev))
        out2, (h_h, c2) = self.R2(out1, (h_h_prev, c_h_prev))
        log_pi, l_t, entropy_mean = self.locator(out2.squeeze(0))
        b_t = self.baseliner(out2.squeeze(0)).squeeze()
        log_probas = self.classifier(out1.squeeze(0))
        return h_t, h_h, c1, c2, l_t, b_t, log_probas, log_pi, None, None, None, None, entropy_mean


class MRAM(nn.Module):
    """MRAM — Multi-Level Recurrent Attention Model (ICONIP).

    Two stacked LSTMs inspired by the human visual pathway. Lower layer (SC-like,
    fast saccadic timescale) drives the location policy; upper layer (visual-
    cortex-like, slow recognition timescale) drives classification. REINFORCE
    baseline is a hybrid MLP over the concatenation of both hidden states
    (eq. 8 of paper). No gating between layers — the lower output is fed
    directly to the upper RNN.
    """

    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes):
        super().__init__()
        self.std = std
        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.R1 = nn.LSTM(hidden_size, hidden_size)
        self.R2 = nn.LSTM(hidden_size, hidden_size)
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size * 2, 1)

    def forward(self, x, l_t_prev, h_t_prev, h_h_prev, c_t_prev, c_h_prev, sigma=None):
        g_t = self.sensor(x, l_t_prev)
        out1, (h_t, c1) = self.R1(g_t.unsqueeze(0), (h_t_prev, c_t_prev))
        out2, (h_h, c2) = self.R2(out1, (h_h_prev, c_h_prev))
        combined = torch.cat([out1.squeeze(0), out2.squeeze(0)], dim=-1)
        log_pi, l_t, entropy_mean = self.locator(out1.squeeze(0))
        b_t = self.baseliner(combined).squeeze()
        log_probas = self.classifier(out2.squeeze(0))
        return h_t, h_h, c1, c2, l_t, b_t, log_probas, log_pi, None, None, None, None, entropy_mean
