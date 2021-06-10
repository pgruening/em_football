from DLBio.train_interfaces import Accuracy, ErrorRate, ITrainInterface
import torch.nn as nn
import torch


def get_interface(ti_type, model, device, printer, **kwargs):
    if ti_type == MatchAndGoals.name:
        return MatchAndGoals(model, device, printer, **kwargs)
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
        self.c0 = float(kwargs.get('c0', [100.0])[0])

    def train_step(self, sample):
        teams, goals = sample[0].to(self.d), sample[1].to(self.d)
        goals1 = goals[:, 0]
        goals2 = goals[:, 1]

        pred = self.model(teams)
        p1 = pred[:, 0]
        p2 = pred[:, 1]

        win = self.c0 * torch.relu(p1 - p2)
        draw = -2. * self.c0 * torch.abs(p1 - p2) + self.c0
        loose = self.c0 * torch.relu(p2 - p1)

        pred = torch.softmax(torch.stack([win, draw, loose], -1), -1)
        # to indices
        targets = torch.stack([
            goals1 > goals2, goals1 == goals2, goals2 > goals1
        ], -1).long().max(-1)[1]

        xent = self.xent_loss(pred, targets)
        mse_1 = self.mse(goals1.float(), p1)
        mse_2 = self.mse(goals2.float(), p2)
        loss = self.alpha * xent + self.beta * .5 * (mse_1 + mse_2)

        abs_1 = torch.abs(goals1.float() - p1).mean().item()
        abs_2 = torch.abs(goals2.float() - p2).mean().item()

        assert not bool(torch.isnan(loss))

        metrics = {
            'mse': .5 * (mse_1 + mse_2).item(), 'xent': xent.item(),
            'abs': .5 * (abs_1 + abs_2)
        }
        counters = None
        functions = {
            k: f.update(pred, targets) for k, f in self.functions.items()
        }
        return loss, metrics, counters, functions
