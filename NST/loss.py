"""
Define the loss function for style loss, content loss, and total_variation loss
"""

def content_loss(content_weight, current,original):
    r"""Return the content loss
    Args:
        content_weight (float): weight for content loss
        current (torch.tensor): current image tensor at specific layer
        original (torch.tensor): original image tensor at specific layer
    """
    if current.shape != original.shape:
        raise ValueError("feature shapes should be identical")
    return content_weight * ((current-original).pow(2)).sum()

def style_loss(layer_weight, current, styler,normalize=True):
    r"""Return the style loss
    Args:
        layer_weight (float): weight for each layer
        current (torch.tensor): current image tensor at specific layer
        styler (torch.tensor): styler tensor at the same specific layer
        normalize (bool): whether to normalize
    """
    N,C,H,W = current.shape
    NN,CC,HH,WW = styler.shape
    if C!=CC:
        raise ValueError('channels number should be identical')

    current = current.view(C,-1)
    styler =styler.view(C,-1)

    gramm_current = current @ current.T
    gramm_styler = styler @ styler.T
    if normalize:
        gramm_current /= float(C*H*W)
        gramm_styler /= float(CC*HH*WW)

    return layer_weight * ((gramm_current-gramm_styler)**2).sum()

def total_variation_reg(reg_weight,current):
    r"""Return the total variation loss
    Args:
        reg_weight (float): weight for total variation regularization
        current (torch.tensor): current image tensor
    """
    loss = 0
    loss += ((current[:,:,1:,:]-current[:,:,:-1,:])**2).sum()
    loss += ((current[:,:,:,1:]-current[:,:,:,:-1])**2).sum()

    return reg_weight * loss
