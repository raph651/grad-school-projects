"""
The main scipt for traning a NST on a content image with selected styler image
Copy from the original image, update the new image with gradient descent in loss,
where loss = content loss + style loss + total variation regularization.

Default epochs to optimize: 400
Default learning rate decay to 0.1, at epoch 350
Default clamp the image tensor to (-1,5,1.5) from epoch 0 to 380
Default show image at every 50 epochs
Default init_noise =False: use the original image as starting point, True for noise
"""
import tqdm
import torch.nn
import matplotlib.pyplot as plt

from dataprep import NSTdata, transform_back
from config import NST_args, prep_args
from loss import content_loss, style_loss, total_variation_reg
from models import Vgg16, Efficientnet_b3

def main(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_param, optim_param = prep_args(params)

    optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
    backbone_dict = {
                'efficientnet_b3': Efficientnet_b3(),
                'vgg16': Vgg16(),
            }

    backbone = backbone_dict[model_param]

    model = backbone.model
    model.to(device)
    ##check the model parameters doesnt require grad
    #for i in model.parameters():
    #    print('require grad: ',i.requires_grad)
    #    break

    data = NSTdata(params.sp, params.ip, params.ss, params.cs)
    styler_inp, img_inp = data.build()

    optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}

    dtype = torch.cuda.FloatTensor # Uncomment this to use GPU
    model.type(dtype)

    styler_features = backbone.extract_features(styler_inp.cuda())
    content_features = backbone.extract_features(img_inp.cuda())

    style_layer = backbone._style_layers()
    content_layer = backbone._content_layers()

    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(transform_back(img_inp.cpu()))
    axarr[1].imshow(transform_back(styler_inp.cpu()))
    plt.show()
    #plt.figure()

    initial_noise = False
    if initial_noise:
        img = torch.Tensor(img_inp.size()).uniform_(0, 1).clone().type(dtype)
    else:
        img =img_inp.clone().type(dtype)

    img.requires_grad= True

    optimizer = optim_dict[optim_param['optim']]([img],**optim_arg)

    decayed_lr = 0.1
    decay_lr_at = 350

    epochs = 400
    plot_interval = 50

    m_ax =5     # subplot (n_ax, m_ax)  m_ax default to be 5
    n_ax = epochs//plot_interval
    n_ax = n_ax//m_ax + int((n_ax%m_ax)>0)
    fig,ax = plt.subplots(n_ax ,m_ax,figsize=(14,7))

    plot_n=0
    for epoch in tqdm.trange(epochs):
        if epoch < 380:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()
        features = backbone.extract_features(img)

        c_loss=0
        s_loss=0

        for idx,layer in enumerate(style_layer):
            s_loss += style_loss(params.sw[idx],features[layer], styler_features[layer])
        for idx,layer in enumerate(content_layer):
            c_loss += content_loss(params.cw, features[layer], content_features[layer])

        loss = c_loss + s_loss + total_variation_reg(params.tvw, img)
        if epoch % 50 ==0: print(loss)
        loss.backward()
        if epoch == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)

        optimizer.step()
        if epoch % 50 == 0:
            ax[plot_n//m_ax][plot_n%m_ax].axis('off')
            ax[plot_n//m_ax][plot_n%m_ax].imshow(transform_back(img.data.cpu()))
            ax[plot_n//m_ax][plot_n%m_ax].set_title(f'Iteration {epoch}')
            #print('Iteration {}'.format(epoch))
            #plt.axis('off')
            #plt.imshow(transform_back(img.data.cpu()))
            #plt.show()
            plot_n+=1

    # last epoch, final pic
    ax[plot_n//m_ax][plot_n%m_ax].axis('off')
    ax[plot_n//m_ax][plot_n%m_ax].imshow(transform_back(img.data.cpu()))
    ax[plot_n//m_ax][plot_n%m_ax].set_title(f'Iteration {epoch+1}')

    for _ in range(plot_n+1,n_ax*m_ax):
        ax[_//m_ax][_%m_ax].set_visible(False)

    fig.tight_layout()
    plt.show()
    #print('Iteration {}'.format(epoch))
    #plt.axis('off')
    #plt.imshow(transform_back(img.data.cpu()))
    #plt.show()

if __name__ == '__main__':

    params = NST_args.parse_args()
    main(params)
