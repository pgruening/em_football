from DLBio.train_interfaces import Accuracy, ErrorRate, ITrainInterface
import torch.nn as nn
import torch


def get_interface(ti_type, model, device, printer, **kwargs):
    if ti_type == MatchAndGoals.name:
        return MatchAndGoals(model, device, printer)
    raise ValueError(f"Unknown ti_type: {ti_type}")


class MatchAndGoals(ITrainInterface):
    name = 'MatchAndGoals'

    def __init__(self, model, device, printer, **kwargs):
        self.printer = printer
        self.model = model
        self.xent_loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.functions = {
            'acc': Accuracy(),
            'er': ErrorRate()
        }
        self.counters = {}
        self.d = device
        self.alpha = float(kwargs.get('alpha', [1.0])[0])
        self.beta = float(kwargs.get('beta', [1.0])[0])

    def train_step(self, sample):
        team1, team2 = sample[0].to(self.d), sample[1].to(self.d)
        goals1, goals2 = sample[2].to(self.d), sample[3].to(self.d)

        p1, p2 = self.model(team1, team2)

        win = 100. * torch.relu(p1 - p2)
        draw = -200. * torch.abs(p1 - p2) + 100.
        loose = 100. * torch.relu(p2 - p1)

        pred = torch.sigmoid(torch.stack([win, draw, loose], -1), -1)
        targets = torch.stack([
            goals1 > 0, goals2, goals1 == goals1, goals2 > goals1
        ], -1).float()

        xent = self.xent_loss(pred, targets)
        mse_1 = self.mse(goals1, p1)
        mse_2 = self.mse(goals1, p2)
        loss = self.alpha * xent + self.beta * .5 * (mse_1 + mse_2)

        assert not bool(torch.isnan(loss))

        metrics = None
        counters = None
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions
