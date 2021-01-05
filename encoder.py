import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def tie_weights_2d1d(src, trg):
    assert type(src) == type(trg)
    trg.spatial_conv.weight = src.spatial_conv.weight
    trg.spatial_conv.bias = src.spatial_conv.bias
    trg.temporal_conv.weight = src.temporal_conv.weight
    trg.temporal_conv.bias = src.temporal_conv.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, 
        channels=[16, 32, 32], 
        num_layers=2, 
        num_filters=32,
        output_logits=False,
        image_channel=3):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.image_channel = image_channel
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.outputs = dict()
        
        x = torch.randn([32]+list(obs_shape))
        self.out_dim = self.forward_conv(x,flatten=False).shape[-1]
        print('conv output dim: ' + str(self.out_dim))
        
        self.fc = nn.Linear(num_filters * self.out_dim * self.out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs,flatten=True):
        if obs.max() > 1.:
            obs = obs / 255.

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
        
        if flatten:
            return conv.view(conv.size(0), -1)
        else:
            return conv
    
    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        
        try:
            h_fc = self.fc(h)
        except:
            print(obs.shape)
            print(h.shape)
            assert False
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class PixelDelta2DEncoder(nn.Module):
    """Flare encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, 
        channels=[16, 32, 32], 
        num_layers=2, 
        num_filters=32,
        output_logits=False,
        image_channel=3):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.image_channel = image_channel

        time_step = obs_shape[0] // self.image_channel

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.image_channel, num_filters, 3, stride=2)]
        )
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        for i in range(2, num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.outputs = dict()
        
        x = torch.randn([32]+list(obs_shape))
        self.out_dim = self.forward_conv(x,flatten=False).shape[-1]

        print('conv output dim: ' + str(self.out_dim))

        self.fc = nn.Linear(num_filters * self.out_dim * self.out_dim * (2*time_step-2), self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits 


    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs,flatten=True):
        if obs.max() > 1.:
            obs = obs / 255.

        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0]*time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        self.outputs['obs'] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv%s' % (1 + 1)] = conv

        for i in range(2, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        conv = conv.view(conv.size(0)//time_step, time_step, conv.size(1), conv.size(2), conv.size(3))

        conv_current = conv[:, 1:, :, :, :]
        conv_prev = conv_current - conv[:, :time_step-1, :, :, :].detach()
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1)*conv.size(2), conv.size(3), conv.size(4))

        if not flatten:
            return conv
        else:
            conv = conv.view(conv.size(0), -1)
            return conv           

    
    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()
        
        try:
            h_fc = self.fc(h)
        except:
            print(obs.shape)
            print(h.shape)
            assert False
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'pixel_delta2d': PixelDelta2DEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, 
    output_logits=False, 
    channels=[16,32,32], 
    image_channel=3,
):
    assert encoder_type in _AVAILABLE_ENCODERS
    if image_channel == 3:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, channels, num_layers, num_filters, output_logits
        )
    else:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, channels, num_layers, num_filters, output_logits, image_channel
        )

