import torch



def content_loss(content_weight, current,original):
    if current.shape != original.shape:
        raise ValueError("feature shapes should be identical")
    return content_weight*torch.sum((current-original).pow(2))
    
def style_loss(layer_weight, current, styler,normalize=True):
    N,C,H,W = current.shape
    NN,CC,HH,WW = styler.shape
    if C!=CC:
        raise ValueError('channels number should be identical')
        
    current= current.view(C,-1)
    styler =styler.view(C,-1)
    
    Gramm_current = torch.mm(current, current.t())
    Gramm_styler = torch.mm(styler,styler.t())
    if normalize:
        Gramm_current /= float(C*H*W)
        Gramm_styler /= float(CC*HH*WW)
    #print(torch.sum((Gramm_current-Gramm_styler)**2))
    
    return layer_weight*torch.sum((Gramm_current-Gramm_styler)**2)

def total_variation_reg(reg_weight,current):
    
    loss=0
    loss+= torch.sum((current[:,:,1:,:]-current[:,:,:-1,:])**2)
    loss+= torch.sum((current[:,:,:,1:]-current[:,:,:,:-1])**2)
    
    return reg_weight*loss
    
    