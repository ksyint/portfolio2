import torch
import torch.nn as nn

class BCE2(nn.Module):
    def __init__(self, penalty_factor=0.1):
        super(BCE2, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.penalty_factor = penalty_factor

    def forward(self, input, target):
        bce_loss_value = self.bce_loss(input, target)
        penalty = torch.where(target == 1.0, self.penalty_factor * bce_loss_value, torch.tensor(0.0))
        custom_loss = bce_loss_value + penalty
        return custom_loss
    

class BCE3(nn.Module):
    def __init__(self, weight_increase_factor=0.15):
        super(BCE3, self).__init__()
        self.weight_increase_factor = weight_increase_factor
        self.bce_loss = nn.BCELoss()

    def forward(self, logits, labels):
        base_loss = self.bce_loss(logits, labels)

        weighted_loss = base_loss + torch.mean(self.weight_increase_factor * labels * base_loss)

        return weighted_loss