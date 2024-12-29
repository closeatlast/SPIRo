import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassificationLoss(nn.Module):
    """
    A thin wrapper around BCEWithLogitsLoss for binary classification.
    Assumes model outputs a single logit per sample.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        # If you have imbalanced data, you can set pos_weight > 1
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, 1) raw output (no sigmoid) from the model
            targets: (batch_size, 1) binary labels in {0,1}

        Returns:
            Scalar loss for backward pass
        """
        loss = self.criterion(logits, targets.float())
        return loss
