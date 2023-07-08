import torch


CONSTANT1 = 10  
CONSTANT2 = 5


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy): # (c_x, c_y, w, h)-> (x_min, y_min, x_max, y_max)
    return torch.cat([
        cxcy[:, :2] - (cxcy[:, 2:] / 2), # x_min, y_min
        cxcy[:, :2] + (cxcy[:, 2:] / 2), # x_max, y_max
    ], dim=1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / CONSTANT1),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * CONSTANT2], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155

    return torch.cat([
        gcxgcy[:, :2] * priors_cxcy[:, 2:] / CONSTANT1 + priors_cxcy[:],
        torch.exp(gcxgcy[:, 2:] / CONSTANT2) * priors_cxcy[:, 2:]
    ], dim=1)


def find_intersection(set1, set2):
    """
    set1 : tensor of dimension (n1, 4)
    set2 : tensor of dimension (n2, 4)
    """

    lower_bounds = torch.max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0)) # (n1, n2, 2)
    upper_bounds = torch.min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0)) # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] # (n1, n2)


def find_jaccard_overlap(set1, set2):
    """
    set1 : tensor of dimension (n1, 4)
    set2 : tensor of dimension (n2, 4)
    """

    intersection = find_intersection(set1, set2)

    # find areas of each box in both sets
    areas_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1]) # (n1)
    areas_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1]) # (n2)

    # find the union
    union = areas_set1.unsqueeze(1) + areas_set2.unsqueeze(0) - intersection # (n1, n2)

    return intersection/union # (n1, n2)




def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor