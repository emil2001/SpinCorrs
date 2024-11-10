import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.transforms.coupling import AffineCouplingTransform, PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.nonlinearities import PiecewiseRationalQuadraticCDF
from nflows.transforms.splines import rational_quadratic_spline


class DeepSet(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size=200, n_hidden_layers=3, dropout=.1, activation_layer=torch.nn.Mish()):
        super(DeepSet, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.activation = activation_layer
        self.sigmoid = torch.nn.Sigmoid()
        self.emb_dim = emb_size
        current_dim = input_size
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(activation_layer)
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, self.emb_dim))
        layers.append(activation_layer)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = self.net(x)
        return x
    
    
class MLP_net(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_hidden_layers, dropout, activation_layer=torch.nn.Mish()):
        super(MLP_net, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.activation = activation_layer
        self.sigmoid = torch.nn.Sigmoid()
        self.emb_dim = emb_size
        current_dim = input_size
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(activation_layer)
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, self.emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
    
    
class Spline_net(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, context_size, encoder, n_hidden_layers=3, dropout=.1, activation_layer=torch.nn.Mish()):
        super().__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.activation = activation_layer
        self.sigmoid = torch.nn.Sigmoid()
        self.emb_dim = emb_size  
        if context_size != 0:
            self.ds = encoder
        current_dim = input_size + context_size 
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(activation_layer)
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, self.emb_dim))
        self.net = nn.Sequential(*layers)


    def forward(self, x, context=None):
        if context != None:
            context = self.ds(context)
            x = torch.cat((x,context), 1)
        x = self.net(x)
        return x

class KAN_net(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, context_size, encoder, n_hidden_layers=3, dropout=.1, activation_layer=torch.nn.Mish()):
        super().__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.activation = activation_layer
        self.sigmoid = torch.nn.Sigmoid()
        self.emb_dim = emb_size  
        if context_size != 0:
            self.ds = encoder
        current_dim = input_size + context_size 
        layers = []
        for i in range(n_hidden_layers):
            layers.append(ekan.KANLinear(current_dim, hidden_size))
            layers.append(nn.Dropout(p=dropout))
            layers.append(activation_layer)
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, self.emb_dim))
        self.net = nn.Sequential(*layers)


    def forward(self, x, context=None):
        if context != None:
            context = self.ds(context)
            x = torch.cat((x,context), 1)
        x = self.net(x)
        return x


class Basic_flow(Flow):
    '''
    A simple normalizing flow with conditioning on the distribution level and affine autoregressive layers
    """
    Parameters:
    
    cond_size (int):
    Number of conditioning variables (X for MLP)
    
    target_size (int):
    Number of target variables
    
    num_layers (int, optional):
    Number of autoregressive layers, default - 5
    
    encoder_config (tuple, optional):
    Configuration of the encoder, (layer size, number of layers, dropout value), default - (200, 3, 0.1)
    
    '''
    def __init__(self, cond_size, target_size, num_layers = 5, encoder_config  = (200, 3, 0.1)):
        enc_layer_size, enc_num_layers, end_dropout = encoder_config
        base_dist = ConditionalDiagonalNormal(shape = [target_size], 
                                              context_encoder = DeepSet(cond_size, 2*target_size, 
                                                                        200, 3, 0.1))
        _base_dist = StandardNormal(shape=[target_size])
        _transforms = []
        for _ in range(num_layers):
            _transforms.append(ReversePermutation(features = target_size))
            _transforms.append(MaskedAffineAutoregressiveTransform(features = target_size, 
                                                                  hidden_features = 128, 
                                                                  context_features = cond_size))
        _transform = CompositeTransform(_transforms)
        super().__init__(
            transform = _transform,
            distribution = _base_dist,
        )

        
class Nu_flow(Flow):
    '''
    Nu-Flows-like network [https://arxiv.org/abs/2207.00664].
    Uses piecewise rational-quadratic coupling transform and LU-decomposed linear layers.
    
    """
    Parameters:
    
    encoder:
    Context embedding function (or NN)
    
    target_size (int):
    Number of target variables
    
    masking_order(list, optional):
    Masking list of target variables with values +1 and -1, default - equal split
    
    num_layers (int, optional):
    Number of flow layers, default - 5
    
    context_size (int, optional):
    Number of conditioning variables, default - 32
    
    spline_conf (tuple, optional):
    Configuration of the coupling network, (layer size, number of layers, dropout value), default - (128, 3, 0.1)
    
    ''' 
    def __init__(self, encoder, target_size, masking_order = None, num_layers = 5, context_size = 32, spline_conf = (128, 3, 0.1)):
        if masking_order is None:
            _mask = torch.ones(target_size)
            _mask[len(_mask)//2:] = - _mask[len(_mask)//2:]
        else:
            _mask = torch.tensor(masking_order)
        _base_dist = StandardNormal(shape=[target_size])
        _transforms = []
        def create_spline_net(input_size, out_size):
            _hidden_size, _n_hidden_layers, _dropout = spline_conf
            return Spline_net(input_size,
                           out_size,
                           hidden_size = _hidden_size,
                           encoder = encoder,
                           n_hidden_layers = _n_hidden_layers,
                           dropout = _dropout,
                           context_size = context_size
                           
            )
        for _ in range(num_layers):
            _transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask = _mask, transform_net_create_fn = create_spline_net, tails = "linear", tail_bound = 4.0))
            _transforms.append(LULinear(features=target_size))
            _mask *= -1
        _transform = CompositeTransform(_transforms)
        super().__init__(
            transform=_transform,
            distribution=_base_dist,
        )
