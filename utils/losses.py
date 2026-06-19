import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss
    TI = (TP + eps) / (TP + alpha_t * FN + beta_t * FP + eps)
    L_FT = (1 - TI)^gamma
    """
    def __init__(self, alpha_t=0.7, beta_t=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(inputs_prob.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        TP = (inputs_flat * targets_flat).sum(1)
        FP = ((1 - targets_flat) * inputs_flat).sum(1)
        FN = (targets_flat * (1 - inputs_flat)).sum(1)
        
        tversky = (TP + self.smooth) / (TP + self.alpha_t * FN + self.beta_t * FP + self.smooth)
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
    Dynamic composite loss for CASS-Net as defined in the manuscript.
    
    L_total(t) = w(t) * L_FT + [1 - w(t)] * L_Dice + lambda(t) * L_aux_mean
    w(t) = weight_start - (weight_start - weight_end) * (t / T)
    """
    def __init__(self, weight_start=0.7, weight_end=0.3, total_epochs=200):
        super().__init__()
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.total_epochs = total_epochs
        
        # Initialize sub-losses (using defaults matching manuscript: alpha_t=0.7, beta_t=0.3, gamma=2.0)
        self.focal_tversky = FocalTverskyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets, epoch):
        # Unpack predictions (Main + 3 Aux)
        if isinstance(preds, tuple):
            pred_main, aux1, aux2, aux3 = preds
        else:
            return self.dice_loss(preds, targets) # Validation/Inference fallback

        # Calculate exact progression to reach the final value at the last epoch
        T = max(self.total_epochs - 1, 1)
        progress = epoch / T

        # Dynamic weighting w(t)
        w_t = self.weight_start - (self.weight_start - self.weight_end) * progress
        w_t = max(self.weight_end, w_t)

        # Main Loss
        ft_loss = self.focal_tversky(pred_main, targets)
        dice_loss = self.dice_loss(pred_main, targets)
        l_main = w_t * ft_loss + (1 - w_t) * dice_loss

        # Auxiliary weighting lambda(t) = 0.5 * (1 - t/T)
        lambda_t = 0.5 * (1 - progress)
        lambda_t = max(0.0, lambda_t)
        
        # L_aux_mean
        l_aux_mean = (self.dice_loss(aux1, targets) + 
                      self.dice_loss(aux2, targets) + 
                      self.dice_loss(aux3, targets)) / 3.0

        return l_main + lambda_t * l_aux_mean
