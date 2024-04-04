import torch


def ibertSoftmax(x):

    coeffA = 1
    coeffB = 7
    coeffC = 24
    log2 = 5
    n_levels = torch.Tensor((256,))
    zero = torch.Tensor((0.,))

    xTilde = (x - torch.max(x, dim=-1, keepdim=True)[0])
    z = torch.floor(-xTilde / log2)
    p = xTilde + z * log2
    y = torch.floor(((coeffA*(p + coeffB)**2 + coeffC)) // (2**z))
    ysum = torch.sum(y, -1, keepdim=True)
    norm = torch.floor(y*(n_levels-1)/(ysum))
    out = torch.clip(norm, zero, n_levels-1)

    return out