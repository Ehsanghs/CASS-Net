import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky loss for binary segmentation."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(probabilities.size(0), -1)
        targets = targets.float().reshape(targets.size(0), -1)

        true_positive = (probabilities * targets).sum(dim=1)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=1)

        tversky_index = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_negative
            + self.beta * false_positive
            + self.smooth
        )

        return ((1.0 - tversky_index) ** self.gamma).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        probabilities = torch.sigmoid(logits).reshape(-1)
        targets = targets.float().reshape(-1)

        intersection = (probabilities * targets).sum()

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice_score


class CASSNetLoss(nn.Module):
    """Dynamic composite loss with deep supervision."""

    def __init__(
        self,
        weight_start=0.7,
        weight_end=0.3,
        auxiliary_weight=0.5,
        total_epochs=200,
        alpha=0.7,
        beta=0.3,
        gamma=2.0,
        smooth=1e-6,
    ):
        super().__init__()

        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero.")

        self.weight_start = float(weight_start)
        self.weight_end = float(weight_end)
        self.auxiliary_weight = float(auxiliary_weight)
        self.total_epochs = int(total_epochs)

        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth,
        )
        self.dice = DiceLoss(smooth=smooth)

    def _progress(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        return min(float(epoch) / float(self.total_epochs), 1.0)

    def _single_output_loss(self, logits, targets, epoch):
        progress = self._progress(epoch)

        weight = self.weight_start - (
            self.weight_start - self.weight_end
        ) * progress

        weight = max(self.weight_end, weight)

        focal_tversky_loss = self.focal_tversky(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            weight * focal_tversky_loss
            + (1.0 - weight) * dice_loss
        )

    def forward(self, predictions, targets, epoch):
        if isinstance(predictions, torch.Tensor):
            return self._single_output_loss(
                predictions,
                targets,
                epoch,
            )

        if not isinstance(predictions, (tuple, list)):
            raise TypeError(
                "predictions must be a tensor, tuple, or list."
            )

        if len(predictions) != 4:
            raise ValueError(
                "Expected main output and three auxiliary outputs."
            )

        main_output, aux1, aux2, aux3 = predictions

        main_loss = self._single_output_loss(
            main_output,
            targets,
            epoch,
        )

        auxiliary_loss = (
            self._single_output_loss(aux1, targets, epoch)
            + self._single_output_loss(aux2, targets, epoch)
            + self._single_output_loss(aux3, targets, epoch)
        ) / 3.0

        progress = self._progress(epoch)
        auxiliary_weight = max(
            0.0,
            self.auxiliary_weight * (1.0 - progress),
        )

        return main_loss + auxiliary_weight * auxiliary_loss
