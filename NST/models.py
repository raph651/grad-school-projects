"""
Setup backbone model, currently only vgg16 and efficientnet_b3 are supported.
Could add other models to extract features.
"""
from torchvision import models
from torch import nn

class Efficientnet_b3(object):
    r"""efficientnet_b3 model
    """
    def __init__(self, feature_extracting=True):
        self.model = models.efficientnet_b3(pretrained=True)
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
        input_size = (320,300) # input size for efficient net b3 -(320,300)

        self.input_size = input_size
        self.layers = None

    def _style_layers(self):
        r"""Return all the names of the feature layers from extract_features function
        """
        if not self.layers: raise ValueError('Extract features first')
        return self.layers  # style layer default to be all the conv layers

    def _content_layers(self):
        r"""Return the name of 1 feature layer from extract_features function
        """
        if not self.layers: raise ValueError('Extract features first')
        return [self.layers[1]]  #content layer default to be the second layer
                                 #(the first MBConv layer)

    def extract_features(self, x, num_seq=3):
        r"""Return features from efficientnet_b3 model.
        Args:
            num_seq (int): number of nn.Sequential that been used for extracting,
                           default to be the first 3
        """
        seq_feature = self.model.features[:num_seq] #first 3 nn.Sequential list

        features = {}
        layer_names= []

        prev_feat = x

        MBConv_count = 1
        for idx, layer_set in enumerate(seq_feature):
            if idx == 0 :
                name = 'ConvActicationSet'
                layer_names.append(name)
                feat = layer_set(prev_feat)
                features[name] = feat
                prev_feat = feat
            else:
                for MBConv in layer_set:
                    name = 'MBConv_' + str(MBConv_count)
                    layer_names.append(name)
                    MBConv_count += 1
                    feat = MBConv(prev_feat)
                    features[name] = feat
                    prev_feat = feat

        self.layers = layer_names # set layers variable for _style_layers, and _content_layers
        return features

class Vgg16(object):
    r"""vgg16 model
    """
    def __init__(self, feature_extracting =True):
        self.model = models.vgg16(pretrained=True)
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
        input_size = 224 # input size for vgg16 -224

        self.input_size =input_size

    def _style_layers(self):
        r"""Return the names of 4 conv layers from vgg16 for computing style loss
        """
        return ['conv_1','conv_4','conv_5','conv_8']
    def _content_layers(self):
        r"""Return the name of 1 conv layer from vgg16 for computing content loss
        """
        return ['conv_2']

    def extract_features(self,x):
        r"""Return features from different layers from vgg16
        """
        features = {}
        prev_feat = x
        for i, module in enumerate(self.model._modules.values()):
            if isinstance(module, nn.Sequential) and i == 0:
                for layer in module.children():
                    if isinstance(layer, nn.Conv2d):
                        i += 1
                        name = 'conv_{}'.format(i)
                    elif isinstance(layer, nn.ReLU):
                        name = 'relu_{}'.format(i)
                        # The in-place version doesn't play very nicely with the ContentLoss
                        # and StyleLoss we insert below. So we replace with out-of-place
                        # ones here.
                        layer = nn.ReLU(inplace=False)
                    elif isinstance(layer, nn.MaxPool2d):
                        name = 'pool_{}'.format(i)
                    elif isinstance(layer, nn.BatchNorm2d):
                        name = 'bn_{}'.format(i)
                    else:
                        raise RuntimeError('Unrecognized layer: {}'.format(
                            layer.__class__.__name__))
                    next_feat = layer(prev_feat)
                    features[name] = next_feat
                    prev_feat = next_feat
        return features
