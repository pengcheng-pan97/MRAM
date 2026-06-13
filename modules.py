"""ICONIP — model components (retina, glimpse network, policy heads)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Retina:
    """Foveated multi-scale glimpse sensor.

    Extracts `k` square patches centred at location `l`. The first patch is
    `g x g`; each subsequent patch covers `s` times the area but is resized
    back to `g x g` so the patches stack into a flat (B, k*g*g*C) tensor.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        phi = []
        size = self.g
        for _ in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        for i in range(1, len(phi)):
            pool = int(phi[i].shape[-1] // self.g)
            phi[i] = F.avg_pool2d(phi[i], pool)

        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)
        return phi

    def extract_patch(self, x, l, size):
        B, C, H, W = x.shape
        start = self.denormalize(H, l)
        end = start + size
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1]: end[i, 1], start[i, 0]: end[i, 0]])
        return torch.stack(patch)

    @staticmethod
    def denormalize(T, coords):
        return (0.5 * ((coords + 1.0) * T)).long()


class GlimpseNetwork(nn.Module):
    """`g_t = relu( fc3(fc1(phi)) + fc4(fc2(l)) )` — combines what + where."""

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()
        self.retina = Retina(g, k, s)
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)
        self.fc2 = nn.Linear(2, h_l)
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev):
        phi = self.retina.foveate(x, l_t_prev)
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))
        what = self.fc3(phi_out)
        where = self.fc4(l_out)
        return F.relu(what + where)


class ActionNetwork(nn.Module):
    """Linear log-softmax classifier on top of the (upper) hidden state."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        return F.log_softmax(self.fc(h_t), dim=1)


class LocationNetwork(nn.Module):
    """Gaussian location policy with FIXED variance `std`.

    Reads the (detached) hidden state, predicts the mean μ via fc + tanh,
    samples ℓ ~ Normal(μ, std²) with the reparameterisation trick.
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()
        self.std = std
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        l_t = torch.distributions.Normal(mu, self.std).rsample().detach()
        log_pi = Normal(mu, self.std).log_prob(l_t).sum(dim=1)
        l_t = torch.clamp(l_t, -1, 1)
        entropy = Normal(mu, self.std).entropy().sum(dim=1).mean().item()
        return log_pi, l_t, entropy


class BaselineNetwork(nn.Module):
    """Linear baseline used for REINFORCE variance reduction. For MRAM the
    input is `concat([h_t1, h_t2])` (hybrid baseline, eq. 8); for RAM/DRAM
    the input is the relevant single hidden state."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        return self.fc(h_t.detach())
