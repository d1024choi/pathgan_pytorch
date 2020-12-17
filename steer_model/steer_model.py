'''
 ENCODER-DECODER BASED RNN MODEL FOR LOCAL PATH GENERATION
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 VERSION : 2020-02-11
'''

from torchvision import models
from utils.functions import *

def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):

    '''
    Make multi-layer perceptron
    example) (dim_list = [32, 64, 16], activation='relu', batch_norm=False, dropout=0)
             will make two fully-connected layers.
             1) fully-connected layer 1 of size (32 x 64), with 'relu' activation, without batch_norm, with dropout prob. 0
             2) fully-connected layer 2 of size (64 x 16), with 'relu' activation, without batch_norm, with dropout prob. 0
    '''

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)

def make_weight_matrix(row, col):

    weight = Parameter(torch.Tensor(row, col))
    return nn.init.kaiming_normal_(weight)


class Encoder(nn.Module):

    def __init__(self, input_dim=2, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.spatial_embedding = make_mlp(dim_list=[input_dim, embedding_dim])
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.out = self.mlp(dim_list=[h_dim, 1], drop_prob=0)


    def mlp(self, dim_list, drop_prob):

        num_layer = len(dim_list) - 1
        cnt_layer = 0
        layers = []
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):

            # fully-connected layer
            layers.append(nn.Linear(dim_in, dim_out))
            cnt_layer += 1

            # activation function
            if (num_layer == cnt_layer):
                do_nothing = 0
            else:
                layers.append(nn.ReLU())

            # drop-out layer
            if drop_prob > 0 and (cnt_layer != num_layer):
                layers.append(nn.Dropout(p=drop_prob))

        return nn.Sequential(*layers)

    def forward(self, obs_traj_rel, path):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = path_rel.size(1)
        # path = torch.cumsum(path_rel, dim=1)
        batch = path.size(1)

        obs_traj_embedding = self.spatial_embedding(path.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)

        speed = torch.sqrt(torch.sum(obs_traj_rel[-1] ** 2, dim=1))
        h = torch.ones(self.num_layers, batch, self.h_dim).cuda()
        c = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        for b in range(batch):
            h[:, b, :] = speed[b] * h[:, b, :]

        state_tuple = (h, c)
        output, state = self.encoder(obs_traj_embedding, state_tuple)

        return F.tanh(self.out(state[0]))

def Load_Overall_Models(args, dtype):


    # conv net
    SteerGen = Encoder(input_dim=args.input_dim,
                       embedding_dim=args.emb_dim_enc,
                       h_dim=args.hidden_dim_enc,
                       num_layers=args.num_layers_enc,
                       dropout=0.0)

    SteerGen.type(dtype)

    # optimizer
    opt = optim.Adam(SteerGen.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    return SteerGen, opt































