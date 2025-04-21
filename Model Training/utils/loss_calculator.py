import torch
import torch.nn.functional as F


def calculate_loss( loc_pred, loc_true ):


    # Get the probability map 
    loc_pred_prob = torch.sigmoid(loc_pred)

    loss_loc = F.mse_loss(loc_pred_prob, loc_true)


    return loss_loc

