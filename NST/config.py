import argparse
import os
_parser_train = argparse.ArgumentParser(description='Configuration for NST')

#general 
_parser_train.add_argument('--sp', type=str, default='./images/styler/starry_night.jpg', help='style path')
_parser_train.add_argument('--ip', type=str, default='./images/tubingen.jpg', help='image path')

_parser_train.add_argument('--lr', type=float, default=1e-4, help='learning rate')
_parser_train.add_argument('--weight_decay', type=float, default=None, help='weight decay')
_parser_train.add_argument('--optim', type=str, default='adam', help='optimizer')
_parser_train.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')



NST_args = _parser_train

def prep_args(params):
        
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


    _display(params,optim_param)

    return optim_param
    
def _display(params,optim_param):
    print('='*40)
    print('General Configuration:')
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
