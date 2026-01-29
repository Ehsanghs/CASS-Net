import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(inputs_prob.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        TP = (inputs_flat * targets_flat).sum(1)
        FP = ((1 - targets_flat) * inputs_flat).sum(1)
        FN = (targets_flat * (1 - inputs_flat)).sum(1)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = (1.0 - tversky)**self.gamma
        return focal_tversky.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice

class CASSNetLoss(nn.Module):
    """
    Eq 1: L_total = alpha(t)L_FT + (1-alpha(t))L_Dice + lambda(t)L_aux
    """
    def __init__(self, alpha_start=0.7, alpha_end=0.3, total_epochs=200):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs
        self.focal_tversky = FocalTverskyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets, epoch):
        # Unpack predictions (Main + 3 Aux)
        if isinstance(preds, tuple):
            pred_main, aux1, aux2, aux3 = preds
        else:
            return self.dice_loss(preds, targets) # Validation/Inference fallback

        # Dynamic Alpha annealing (Eq. 2)
        alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (epoch / self.total_epochs)
        alpha = max(self.alpha_end, alpha)

        # Main Loss
        ft_loss = self.focal_tversky(pred_main, targets)
        dice_loss = self.dice_loss(pred_main, targets)
        l_main = alpha * ft_loss + (1 - alpha) * dice_loss

        # Aux weighting (Eq. 5)
        # lambda = 0.5 * (1 - t/T)
        lam = 0.5 * (1 - epoch / self.total_epochs)
        lam = max(0.0, lam)
        
        l_aux = (self.dice_loss(aux1, targets) + 
                 self.dice_loss(aux2, targets) + 
                 self.dice_loss(aux3, targets)) / 3.0

        return l_main + lam * l_aux