class TwoClassCrossEntropyLoss(nn.Module):
    """
    Uses CrossEntropyLoss for a 2-logit output.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, 2) raw output from the model
            targets: (batch_size,) class labels in {0,1}

        Returns:
            Scalar loss
        """
        loss = self.criterion(logits, targets)
        return loss
