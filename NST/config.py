"""
Config parameters for NST trainnig
"""
import argparse
import os
_parser_train = argparse.ArgumentParser(description='Configuration for NST')

#general
_parser_train.add_argument('--sp', type=str, default='./images/styler/starry_night.jpg',
                           help='style path')
_parser_train.add_argument('--ip', type=str, default='./images/tubingen.jpg', help='image path')
_parser_train.add_argument('--cw', type=float, default=5e-2, help='content weight')
_parser_train.add_argument('--sw', type=float,  nargs='+', default = [1000,1000,1000,1000],
                           help='style weight in each layer')
_parser_train.add_argument('--tvw', type=float, default=5e-2, help='total variation loss weight')

_parser_train.add_argument('--ss', type=int, nargs='+', default=512, help='style image size')
_parser_train.add_argument('--cs', type=int, nargs='+', default=192, help='content image size')

_parser_train.add_argument('--lr', type=float, default=1e-4, help='learning rate')
_parser_train.add_argument('--weight_decay', type=float, default=0, help='weight decay')
_parser_train.add_argument('--optim', type=str, default='adam', help='optimizer')
_parser_train.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')

_parser_train.add_argument('--backbone', type=str, default='vgg16',
                           help='backbone choice: vgg16, efficientnet_b3')


NST_args = _parser_train

def prep_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and backbone model
    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    if not os.path.isfile(params.ip):
        raise ValueError('image file doesnt exist')
    if not os.path.isfile(params.sp):
        raise ValueError('styler file doesnt exist')

    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay}
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr,
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))

    if params.backbone in ['vgg16', 'efficientnet_b3']:
        model_param = params.backbone
    else:
        raise ValueError('No support backbone model {}'.format(params.backbone))

    #params.ss = tuple(params.ss)
    #params.cs = tuple(params.cs)

    if isinstance(params.ss,list):
        if len(params.ss) == 1:params.ss=int(params.ss[0])
        else:params.ss = tuple(params.ss)

    if isinstance(params.cs,list):
        if len(params.cs) == 1:params.cs=int(params.cs[0])
        else:params.cs = tuple(params.cs)

    if isinstance(params.sw,list):
        if len(params.sw) == 1:
            params.sw = tuple(params.sw*4)
        else:params.sw = tuple(params.sw)

    _display(params,optim_param)

    return model_param, optim_param

def _display(params,optim_param):
    print('='*40)
    print('General Configuration:')
    print(f'Backbone model: {params.backbone}' )
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    print(f'content weight: {params.cw}')
    print(f'style weight in each layer: {params.sw}')
    print(f'total variation loss weight: {params.tvw}')
    print(f'styler image size: {params.ss}')
    print(f'content image size: {params.cs}')
    print('='*40)
