import torch
from torch.nn import functional as F

def dice_loss(output, target, exclude_0=False, weights=None, ignore_index=None, n_classes=2):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1
    output = output.exp()  # to convert log(softmax) output from model to softmax

    encoded_target = output.detach() * 0
    # if ignore_index is not None:
    #   mask = target == ignore_index
    #   target = target.clone()
    #   target[mask] = 0
    #   encoded_target.scatter_(1, target.unsqueeze(1), 1)
    #   mask = mask.unsqueeze(1).expand_as(encoded_target)
    #   encoded_target[mask] = 0
    # else:
    encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if exclude_0 is True:
      encoded_target = encoded_target[:,1:,:,:]
      output = output[:,1:,:,:]

    if weights is None:
        weights = 1

    intersection = output * encoded_target
    numerator = (2 * intersection.sum(2).sum(2)) + smooth 
    denominator = output + encoded_target

    # if ignore_index is not None:
    #     denominator[mask] = 0
    denominator = denominator.sum(2).sum(2) + smooth

    loss_per_slice = 1 - (numerator / denominator)
    loss_per_batch = (loss_per_slice.sum(0))/output.size(0)

    loss_per_channel = weights * loss_per_batch

    return loss_per_channel.sum() / output.size(1)


def focal_tversky_loss(output, target, exclude_0=False, weights=None, alpha=0.7, focal_gamma=0.75):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1
    output = output.exp()  # to convert log(softmax) output from model to softmax

    encoded_target = output.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if exclude_0 is True:
      encoded_target = encoded_target[:,1:,:,:]
      output = output[:,1:,:,:]

    if weights is None:
        weights = 1

    TP = output * encoded_target
    FP = output * (1 - encoded_target)
    FN = (1 - output) * (encoded_target)

    numerator = (TP.sum(2).sum(2)) + smooth
    denominator = (TP + alpha*FN + (1-alpha)*FP)

    denominator = denominator.sum(2).sum(2) + smooth

    loss_per_slice = 1 - (numerator / denominator)
    loss_per_batch = (loss_per_slice.sum(0))/output.size(0)

    focal_gamma_tensor = torch.zeros(output.size(1), device=output.device)

    if exclude_0 is False:
      focal_gamma_tensor[0] = 1
      focal_gamma_tensor[1:] = focal_gamma
    else:
      focal_gamma_tensor[:] = focal_gamma

    loss_per_batch = torch.pow(loss_per_batch, focal_gamma_tensor)
    loss_per_channel = weights * loss_per_batch

    loss_total = loss_per_channel.sum() / output.size(1)

    return loss_total

def focal_tversky_loss_deep_supervised(output, target, exclude_0=False, weights=None, alpha=0.7, focal_gamma=0.75):
    """
    output :[NxCxHxW,....] Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """

    loss_total = 0
    gt1 = target[:,::8,::8]
    gt2 = target[:,::4,::4]
    gt3 = target[:,::2,::2]
    gt4 = target
    modified_target = [gt1,gt2,gt3,gt4]
    output_ = output.copy()
    for i in range(4):

        smooth = 1
        output_[i] = output_[i].exp()  # to convert log(softmax) output from model to softmax

        encoded_target = output_[i].detach() * 0
        encoded_target.scatter_(1, modified_target[i].unsqueeze(1), 1)  # one hot encoding

        if exclude_0 is True:
          encoded_target = encoded_target[:,1:,:,:]
          output_[i] = output_[i][:,1:,:,:]

        if weights is None:
            weights = 1

        TP = output_[i] * encoded_target
        FP = output_[i] * (1 - encoded_target)
        FN = (1 - output_[i]) * (encoded_target)

        numerator = (TP.sum(2).sum(2)) + smooth
        denominator = (TP + alpha*FN + (1-alpha)*FP)

        denominator = denominator.sum(2).sum(2) + smooth
        

        loss_per_slice = 1 - (numerator / denominator)
        loss_per_batch = (loss_per_slice.sum(0))/output_[i].size(0)

        focal_gamma_tensor = torch.zeros(output_[i].size(1), device=output_[i].device)

        if exclude_0 is False:
          focal_gamma_tensor[0] = 1
          focal_gamma_tensor[1:] = focal_gamma
        else:
          focal_gamma_tensor[:] = focal_gamma

        if i != 3:
        	loss_per_batch = torch.pow(loss_per_batch, focal_gamma_tensor)
            
        loss_per_channel = weights * loss_per_batch

        loss_total += loss_per_channel.sum() / output_[i].size(1)

    return loss_total



def jaccard_loss(output, target, exclude_0 = False, weights=None, ignore_index=None, n_classes=2):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1

    output = output.exp()  # to convert log(softmax) output from model to softmax

    encoded_target = output.detach() * 0
    if ignore_index is not None:
      mask = target == ignore_index
      target = target.clone()
      target[mask] = 0
      encoded_target.scatter_(1, target.unsqueeze(1), 1)
      mask = mask.unsqueeze(1).expand_as(encoded_target)
      encoded_target[mask] = 0
    else:
      encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if weights is None:
        weights = 1

    if exclude_0 is True:
      encoded_target = encoded_target[:,1:,:,:]
      output = output[:,1:,:,:]


    intersection = output * encoded_target
    numerator = (2 * intersection.sum(2).sum(2)) + smooth 
    denominator = output*output + encoded_target*encoded_target

    if ignore_index is not None:
        denominator[mask] = 0
    denominator = denominator.sum(2).sum(2) + smooth

    loss_per_slice = 1 - (numerator / denominator)
    loss_per_batch = (loss_per_slice.sum(0))/output.size(0)

    loss_per_channel = weights * loss_per_batch

    return loss_per_channel.sum() / output.size(1)

def dice_score(output, target, exclude_0 = False, weights=None, ignore_index=None, n_classes=2):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1
    output = output.exp()

    encoded_target = output.detach() * 0
    if ignore_index is not None:
      mask = target == ignore_index
      target = target.clone()
      target[mask] = 0
      encoded_target.scatter_(1, target.unsqueeze(1), 1)
      mask = mask.unsqueeze(1).expand_as(encoded_target)
      encoded_target[mask] = 0
    else:
      encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if weights is None:
        weights = 1

    if exclude_0 is True:
      encoded_target = encoded_target[:,1:,:,:]
      output = output[:,1:,:,:]


    intersection = output * encoded_target
    numerator = (2 * intersection.sum(2).sum(2)) + smooth   #This for 2d slices, if for a patient intersection.sum(0).sum(1).sum(1)
    denominator = output + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0
    denominator = denominator.sum(2).sum(2) + smooth

    score_per_slice = numerator / denominator
    score_per_batch = (score_per_slice.sum(0))/output.size(0)

    score_per_channel = weights * score_per_batch
    return score_per_channel
