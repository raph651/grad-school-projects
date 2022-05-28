import tqdm
import torch.nn

from torchvision import models


from dataprep import NST_data, transform_back
from config import NST_args, prep_args
from loss import content_loss, style_loss, total_variation_reg

class backbone_model(nn.Module):
    


def main(params):
    
    optim_param = prep_args(params)
    optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
    
    data = NST_data(params.sp,params.ip)
    styler_inp, image_inp = data.build()
    
    
    img_inp.require_grad()
    
    optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
    optimizer = optim_dict[optim_param['optim']](img_inp,**optim_arg)
    
    for epoch in tqdm.trange(epochs):
        
    
    
    
    #print(content_loss(image_inp,image_inp))
    #print(style_loss(0.1,styler_inp[0],image_inp[0]))
    
    #print(total_variation_reg(0.01,image_inp))
          
    
    
    
    
if __name__ == '__main__':
    
    params = NST_args.parse_args()
    main(params)