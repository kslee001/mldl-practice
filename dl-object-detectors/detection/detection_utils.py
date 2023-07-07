import torch




def cxcy_to_xy(cxcy): # (c_x, c_y, w, h)-> (x_min, y_min, x_max, y_max)
    return torch.cat([
        cxcy[:, :2] - (cxcy[:, 2:] / 2), # x_min, y_min
        cxcy[:, :2] + (cxcy[:, 2:] / 2), # x_max, y_max
    ], dim=1)



def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155

    CONSTANT1 = 10  
    CONSTANT2 = 5

    return torch.cat([
        gcxgcy[:, :2] * priors_cxcy[:, 2:] / CONSTANT1 + priors_cxcy[:],
        torch.exp(gcxgcy[:, 2:] / CONSTANT2) * priors_cxcy[:, 2:]
    ], dim=1)