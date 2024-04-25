import numpy as np
import torch
import torch.nn as nn
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.base import CompositeTransform

from normalizing_flows.normal import MultivariateNormal
from pytorch_lightning import LightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NWP_encoder_CNN(LightningModule):
    """
    CNN Encoder for NWP data
    runs N_timesteps times to output N_encod time series of the right length
    Encoded time series are stacked to ground stations 
    """
    def __init__(self, hparams, input_shape_nwp, input_channels_nwp, cov_pred_params = None):
        super(NWP_encoder_CNN, self).__init__()
        self.name = 'AE_NWP'
        self.n_layers = hparams['n_layers']
        self.in_channels = hparams['in_channels']
        self.in_channels[0] = input_channels_nwp
        self.out_channels = hparams['out_channels']
        self.kernel_size = hparams['kernel_size']
        self.n_pool = self.n_layers
        self.pool_size = hparams['pool_size']

        if isinstance(hparams['kernel_size'], int):
            self.kernel_size = []
            for i in range(self.n_layers):
                self.kernel_size.append(hparams['kernel_size'])

        if isinstance(hparams['pool_size'], int):
            self.pool_size = []
            for i in range(self.n_layers):
                self.pool_size.append(hparams['pool_size'])

        self.activation_function = hparams['activation']
        self.n_fc = len(hparams['fc_size'])
        self.fc_size = hparams['fc_size']
        if cov_pred_params is None: 
            self.n_encod = hparams['n_encod']
        else: 
            self.n_encod = cov_pred_params['out_encod_nwp']
        self.n_timesteps = hparams['n_timesteps']
        self.struct = hparams['struct']
        n1 = input_shape_nwp[0]
        n2 = input_shape_nwp[1]
        for i in range(self.n_layers):
            n1 = int((n1 - self.kernel_size[i] + 1)/self.pool_size[i])
            n2 = int((n2 - self.kernel_size[i] + 1)/self.pool_size[i])
        self.fc_size[0] = n1*n2*self.out_channels[int(self.n_layers-1)]
        self.conv = nn.ModuleList()
        for i in range(self.n_layers):
            self.conv.append(nn.Conv2d(self.in_channels[i], self.out_channels[i], self.kernel_size[i], stride = 1, padding = 0))
        self.pool = nn.ModuleList()
        for i in range(self.n_pool):
            self.pool.append(nn.MaxPool2d(self.pool_size[i]))

        self.fc = nn.ModuleList()
        for i in range(self.n_fc - 1):
            self.fc.append(nn.Linear(self.fc_size[i], self.fc_size[i+1]))
        self.fc.append(nn.Linear(self.fc_size[-1], self.n_encod*self.n_timesteps))

        if self.activation_function == 'relu':
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(hparams['dropout'])

        self.struct = []
        for i in range(self.n_layers):
            self.struct.append('conv')
            self.struct.append('pool')
        self.struct.append('dropout')
        self.struct.append('flatten')
        self.struct.append('fc')




    def forward(self, x):
        out = torch.zeros((x.shape[0], self.n_encod, self.n_timesteps)).to(x.device)
        x = x.float()
        x_in = x.permute(0, 2, 3, 1, 4)
        x_in = x_in.contiguous().view(x_in.shape[0], x_in.shape[1], x_in.shape[2], -1)
        x_in = x_in.permute(0, 3, 1, 2)
        i_conv = 0
        i_pool = 0
        i_fc = 0
        for i, layer in enumerate(self.struct):
            if layer == 'conv':
                x_in = self.conv[i_conv](x_in)
                x_in = self.activation(x_in)
                # x_in = self.dropout[i_dropout](x_in)
                x_in = self.dropout(x_in)
                # i_dropout += 1
                i_conv += 1
            elif layer == 'pool':
                x_in = self.pool[i_pool](x_in)
                i_pool +=1
            elif layer == 'fc':
                x_in = self.fc[i_fc](x_in)
                if i_fc < len(self.fc):
                    x_in = self.activation(x_in)
                i_fc += 1
            elif layer == 'flatten':
                x_in = torch.flatten(x_in, 1)
            elif layer == 'dropout':
                x_in = self.dropout(x_in)
                
        
        out = x_in.reshape((x_in.shape[0], self.n_encod, self.n_timesteps))

        return out
        

class GS_encoder_CNN(LightningModule):
    """
        CNN for multivariate to multisteps forecasting
        channel = time series
        Convolve in time for smoothing
    """

    def __init__(self, hparams, input_channels_gs, cov_pred_params = None):
        super(GS_encoder_CNN, self).__init__()
        self.name = 'CNN'
        self.n_layers = hparams['n_layers']
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.kernel_size = hparams['kernel_size']
        self.n_pool = self.n_layers
        self.pool_size = hparams['pool_size']

        if isinstance(hparams['kernel_size'], int):
            self.kernel_size = []
            for i in range(self.n_layers):
                self.kernel_size.append(hparams['kernel_size'])

        if isinstance(hparams['pool_size'], int):
            self.pool_size = []
            for i in range(self.n_layers):
                self.pool_size.append(hparams['pool_size'])
        
        self.activation_function = hparams['activation']
        self.n_fc = hparams['n_fc']
        self.fc_size = hparams['fc_size']
        self.out_timesteps = hparams['out_timesteps']
        self.in_timesteps = hparams['in_timesteps']
        if cov_pred_params is None: 
            self.out_encod_gs = hparams['out_encod_gs']
        else: 
            self.out_encod_gs = cov_pred_params['out_encod_gs']
        
        self.struct = hparams['struct']        
        self.in_channels[0] = input_channels_gs
        n1 = self.in_timesteps
        for i in range(self.n_layers):
            n1 = int((n1 - self.kernel_size[i] + 1)/self.pool_size[i])
        
        self.fc_size[0] = n1*self.out_channels[int(self.n_layers-1)]
        self.conv = nn.ModuleList()
        for i in range(self.n_layers):
            self.conv.append(nn.Conv1d(self.in_channels[i], self.out_channels[i], self.kernel_size[i], stride = 1, padding = 0))
        self.pool = nn.ModuleList()
        for i in range(self.n_pool):
            self.pool.append(nn.MaxPool1d(self.pool_size[i]))

        self.fc = nn.ModuleList()
        for i in range(self.n_fc - 1):
            self.fc.append(nn.Linear(self.fc_size[i], self.fc_size[i+1]))
        self.fc.append(nn.Linear(self.fc_size[-1], self.out_timesteps*self.out_encod_gs))

        if self.activation_function == 'relu':
            self.activation = nn.ReLU()
    
        self.dropout = nn.Dropout(hparams['dropout'])

        self.struct = []
        for i in range(self.n_layers):
            self.struct.append('conv')
            self.struct.append('pool')
        self.struct.append('dropout')
        self.struct.append('flatten')
        self.struct.append('fc')


    def forward(self, x):
        x = x.float()
        i_conv = 0
        i_pool = 0
        i_fc = 0
        for i, layer in enumerate(self.struct):
            if layer == 'conv':
                x = self.conv[i_conv](x)
                x = self.activation(x)
                x = self.dropout(x)
                i_conv += 1
            elif layer == 'pool':
                x = self.pool[i_pool](x)
                i_pool +=1
            elif layer == 'fc':
                x = self.fc[i_fc](x)
                i_fc += 1
                if i_fc < len(self.fc):
                    x = self.activation(x)
            elif layer == 'flatten':
                x = torch.flatten(x, 1)
            elif layer == 'dropout':
                x = self.dropout(x)
        
        out = x.reshape(x.shape[0], self.out_encod_gs, self.out_timesteps)
        return out

class FC_mixer(LightningModule):
    def __init__(self, hparams, input_channels_mixer, pred_type = 'mixed', 
                 cov_description = 'pearson', wwpred = False):
        super(FC_mixer, self).__init__()
        self.name = 'FC_mixer'
        self.n_layers = hparams['n_layers']
        self.activation_function = hparams['activation']
        self.fc_size = hparams['fc_size']
        self.n_timesteps = hparams['n_timesteps']
        self.n_out_vars = hparams['n_out_vars'] + wwpred*1
        self.struct = hparams['struct']
        self.pred_type = pred_type
        self.wwpred = wwpred
        self.cov_description = cov_description
        self.fc = nn.ModuleList()
        self.fc_size[0] = input_channels_mixer
        self.fc_size = self.fc_size[:self.n_layers]

        for i in range(self.n_layers - 1):
            self.fc.append(nn.Linear(self.fc_size[i], self.fc_size[i+1]))
        self.fc.append(nn.Linear(self.fc_size[-1], self.n_timesteps*self.n_out_vars))

        if self.activation_function == 'relu':
            self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(hparams['dropout'])
        self.struct = []
        for i in range(self.n_layers-1):
            self.struct.append('fc')
            self.struct.append('dropout')
        self.struct.append('fc')

    def forward(self, x):
        x = x.float()
        # print('Forward pass FC mixer', x.get_device())
        i_fc = 0
        x = torch.flatten(x, 1)
        for i, layer in enumerate(self.struct):
            if layer == 'dropout':
                x = self.dropout(x)
            else:
                x = self.fc[i_fc](x)
                i_fc += 1
                if i_fc < len(self.fc):
                    x = self.activation(x)
        out = x.reshape(x.shape[0], self.n_out_vars, self.n_timesteps)
        if self.pred_type == 'mixed':
            out[:, 2:4, :] = torch.exp(out[:, 2:4, :])
            if self.cov_description == 'pearson':
                out[:, 4, :] = torch.tanh(out[:, 4, :])
            if self.wwpred:
                activ = torch.nn.Sigmoid()
                out[:, 5, :] = activ(out[:, 5, :])
        if self.pred_type == 'cov':
            out[:, :2, :] = torch.exp(out[:, :2, :])
            if self.cov_description == 'pearson':
                out[:, 4, :] = torch.tanh(out[:, 4, :])
        return out

class NF_block(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.NF_params['num_layers']
        self.dimensions = cfg.NF_params['dimensions']
        self.context_model = nn.Identity()
        self.base_dist = MultivariateNormal([2], context_encoder=self.context_model)
        self.transforms = []

        for _ in range(self.num_layers):
            self.transforms.append(ReversePermutation(features=self.dimensions))
            self.transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=self.dimensions, 
                                                                                           hidden_features=cfg.NF_params['hidden_features'], 
                                                                                           context_features = cfg.NF_params['context_features'], 
                                                                                           num_bins = cfg.NF_params['num_bins'], 
                                                                                           tails = cfg.NF_params['tails'],
                                                                                           dropout_probability=cfg.NF_params['dropout']))
        self.transform = CompositeTransform(self.transforms)
        self.flow = Flow(self.transform, self.base_dist, embedding_net = None)

    def sample(self, z):
        samples = []
        with torch.no_grad():
            for i in range(z.shape[0]):
                sample = self.flow.sample(1000, context = z[i, :].float()).numpy()
                sample = sample.squeeze()
                samples.append(sample)
        return samples

    def log_prob(self, z, y):
        loss =  -self.flow.log_prob(inputs = y, context = z)
        return loss.mean(), loss


class CNN1D_baseline_input(LightningModule):
    """
    1D CNN for baseline input Deep Learning model
    """
    def __init__(self, 
                 cfg, 
                 in_channels, 
                 kernel_size, 
                 n_layers, 
                 pool_size, 
                 n_fc, 
                 fc_size, 
                 out_timesteps, 
                 struct, 
                 out_channels, 
                 dropout):
        super(CNN1D_baseline_input, self).__init__()
        self.name = 'CNN1D_baseline'
        self.n_layers = int((np.array(struct) == 'conv').sum())

        self.pred_type = 'mixed'
        self.kernel_size = kernel_size
        self.n_pool = self.n_layers
        self.pool_size = pool_size
        
        if isinstance(kernel_size, int):
            self.kernel_size = []
            for i in range(self.n_layers):
                self.kernel_size.append(kernel_size)

        if isinstance(pool_size, int):
            self.pool_size = []
            for i in range(self.n_layers):
                self.pool_size.append(pool_size)
        self.activation_function = 'relu'
        self.n_fc = n_fc
        self.fc_size = fc_size
        self.out_timesteps = out_timesteps
        
        self.struct = struct
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        #Output is directly the multivariate gaussian
        self.out_encod_gs = 5

        #Compute the size of the CNN output
        n1 = self.out_timesteps
        for i in range(self.n_layers):
            n1 = int((n1 - self.kernel_size[i] + 1)/self.pool_size[i])
        self.fc_size[0] = n1*self.out_channels[int(self.n_layers-1)]
        self.fc_size = self.fc_size[:n_fc]
        #Add convolutional layers
        self.conv = nn.ModuleList()
        for i in range(self.n_layers):
            self.conv.append(nn.Conv1d(self.in_channels[i], self.out_channels[i], self.kernel_size[i], stride = 1, padding = 0))
        self.pool = nn.ModuleList()
        for i in range(self.n_pool):
            self.pool.append(nn.MaxPool1d(self.pool_size[i]))

        #Fully Connected Layers
        self.fc = nn.ModuleList()
        for i in range(self.n_fc - 1):
            self.fc.append(nn.Linear(self.fc_size[i], self.fc_size[i+1]))
        self.fc.append(nn.Linear(self.fc_size[-1], self.out_timesteps*self.out_encod_gs))

        if self.activation_function == 'relu':
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(self.dropout)
        self.NF_transform = cfg.NF_params['NF_transform']
        if self.NF_transform:
            self.NF = NF_block(cfg)

        self.struct = []
        for i in range(self.n_layers):
            self.struct.append('conv')
            self.struct.append('pool')
        self.struct.append('dropout')
        self.struct.append('flatten')
        for i in range(self.n_fc):
            self.struct.append('fc')
        
    def forward(self, x):
        x = x.float()
        i_conv = 0
        i_pool = 0
        i_fc = 0
        out = torch.zeros(x.shape[0], self.out_encod_gs, self.out_timesteps)
        for i, layer in enumerate(self.struct):
            if layer == 'conv':
                x = self.conv[i_conv](x)
                x = self.activation(x)
                x = self.dropout(x)
                i_conv += 1
            elif layer == 'pool':
                x = self.pool[i_pool](x)
                i_pool +=1
            elif layer == 'fc':
                x = self.fc[i_fc](x)
                i_fc += 1
                if i_fc < len(self.fc):
                    x = self.activation(x)
            elif layer == 'flatten':
                x = torch.flatten(x, 1)
            elif layer == 'dropout':
                x = self.dropout(x)
        
        out = x.reshape(x.shape[0], self.out_encod_gs, self.out_timesteps)
        
        out[:, 2:4, :] = torch.exp(out[:, 2:4, :])
        out[:, -1, :] = torch.tanh(out[:, -1, :])

        return out


class MNet_forecast_model_CNN(LightningModule):
    def __init__(self, 
                 cfg, 
                 closest_pred = None, 
                 pred_type = 'mixed', 
                 closest_input_encod = False,
                 **kwargs):
        super(MNet_forecast_model_CNN, self).__init__()
        input_channels_gs = kwargs['input_channels_gs']
        input_shape_nwp = kwargs['input_shape_nwp'] 
        input_channels_nwp = kwargs['input_channels_nwp']
        input_channels_mixer = kwargs['input_channels_mixer_mean']
        self.wwpred = cfg.learning_params.wwpred
        self.NWP_E = NWP_encoder_CNN(cfg.encoder_params, 
                                     input_shape_nwp, 
                                     input_channels_nwp)
        self.GS_E = GS_encoder_CNN(cfg.model_params, 
                                   input_channels_gs) 
        self.pred_type = pred_type
        if pred_type == 'mixed':
            self.mixer_model = FC_mixer(cfg.FC_mixer_params, 
                                        input_channels_mixer, 
                                        pred_type = self.pred_type, 
                                        cov_description=cfg.learning_params['cov_description'], 
                                        wwpred = self.wwpred)
        elif pred_type == 'mean':
            self.mixer_model = FC_mixer(cfg.FC_mixer_mean_params, input_channels_mixer, pred_type = self.pred_type, cov_description=cfg.learning_params['cov_description'])
        self.name = 'MNet_forecast'
        self.closest_pred = closest_pred
        self.closest_input_encod = closest_input_encod
        self.input_gs  = cfg.learning_params['input_gs']
        self.input_nwp  = cfg.learning_params['input_nwp']

        self.NF_transform = cfg.NF_params['NF_transform']
        if self.NF_transform:
            self.NF = NF_block(cfg)


    def forward(self, x):
        if self.input_nwp:
            x_nwp = x['nwp']
            x_in = self.NWP_E(x_nwp)
            if self.input_gs:
                x_gs = x['gs']
                x_encod_gs = self.GS_E(x_gs)
                x_in = torch.concat((x_in, x_encod_gs), 1)
        elif self.input_gs:
            x_gs = x['gs']
            x_in = self.GS_E(x_gs)

        if self.closest_input_encod:
            x_encod_closest = x['closest']
            x_in = torch.concat((x_in, x_encod_closest), 1)
        out = self.mixer_model(x_in)
        return out