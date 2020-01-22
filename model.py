import torch.nn as nn
from torch.distributions import Categorical#, Normal

class DiscretePolicy(nn.Module):
    def __init__(self, n_obs, n_acts, n_hidden):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_acts),
        )

    def get_distribution(self, s):
        logits = self.layers(s)
        return Categorical(logits=logits)

    def log_prob(self, s, a):
        return self.get_distribution(s).log_prob(a)

    def forward(self, s):
        return self.get_distribution(s).sample()


class Value(nn.Module):
    def __init__(self, n_obs, n_hidden):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_obs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        
    def forward(self, x):
        return self.layers(x)