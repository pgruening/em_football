import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_dim, feat_dim):
        self.one_hot_to_feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        self.feature_to_goals = nn.Sequential(
            nn.Linear(3 * feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x1, x2):
        x1 = self.one_hot_to_feature(x1)
        x2 = self.one_hot_to_feature(x2)
        rand_z = torch.rand(x1.shape)

        x = torch.cat([x1, x2, rand_z], -1)
        return self.feature_to_goals(x)
