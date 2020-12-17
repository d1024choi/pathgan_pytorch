from torchvision import models
from utils.functions import *

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

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

class ConvNet(nn.Module):

    def __init__(self, use_pretrained=True, feature_extract=False, resent_model=18):
        super(ConvNet, self).__init__()

        if (resent_model == 18):
            self.model_ft = nn.Sequential(*list(models.resnet18(pretrained=use_pretrained).children())[:-2])
        elif(resent_model == 34):
            self.model_ft = nn.Sequential(*list(models.resnet34(pretrained=use_pretrained).children())[:-2])
        elif(resent_model == 50):
            self.model_ft = nn.Sequential(*list(models.resnet50(pretrained=use_pretrained).children())[:-2])
        elif(resent_model == 101):
            self.model_ft = nn.Sequential(*list(models.resnet101(pretrained=use_pretrained).children())[:-2])
        elif(resent_model == 152):
            self.model_ft = nn.Sequential(*list(models.resnet152(pretrained=use_pretrained).children())[:-2])

        self.set_parameter_requires_grad(self.model_ft, feature_extract)

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, img_batch):

        '''
        img_batch : batch size x ch x row x col
        '''

        conv_out = self.model_ft(img_batch)
        batch = conv_out.size(0)
        ch = conv_out.size(1)

        return conv_out.view(batch, ch, -1).permute(0, 2, 1).contiguous()

class PathGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 obs_len,
                 pred_len,
                 emb_dim_dec,
                 hid_dim_dec,
                 num_layers_dec,
                 onehot_dim,
                 intend_dim,
                 conv_flat_size,
                 noise_dim,
                 dropout):
        super(PathGenerator, self).__init__()



        self.input_dim = input_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.emb_dim_dec = emb_dim_dec
        self.h_dim_dec = hid_dim_dec
        self.num_layers_dec = num_layers_dec
        self.onehot_dim = onehot_dim
        self.intend_dim = intend_dim
        self.conv_flat_size = conv_flat_size
        self.noise_dim = noise_dim
        self.dropout = dropout


        # encoder
        self.intention_embedding = make_mlp(dim_list=[self.onehot_dim, self.intend_dim])
        self.mixing_mlp = make_mlp(dim_list=[self.noise_dim+self.intend_dim, self.h_dim_dec])
        self.conv_embedding = make_mlp(dim_list=[self.conv_flat_size, self.h_dim_dec])

        # decoder
        self.decoder = nn.LSTM(emb_dim_dec, hid_dim_dec, num_layers_dec, dropout=dropout)
        self.spatial_embedding = make_mlp(dim_list=[input_dim+onehot_dim, emb_dim_dec])
        self.spatial_embedding_att = make_mlp(dim_list=[emb_dim_dec+hid_dim_dec, emb_dim_dec])
        self.hidden2pos = nn.Linear(hid_dim_dec, input_dim)

        self.spatial_embedding_drvint = make_mlp(dim_list=[emb_dim_dec + hid_dim_dec + input_dim + onehot_dim, emb_dim_dec])
        self.drvsts_decoder = nn.LSTM(emb_dim_dec, hid_dim_dec, num_layers_dec, dropout=dropout)
        self.hidden2class = nn.Linear(hid_dim_dec, onehot_dim)

        self.Tanh = nn.Tanh()
        self.Wh = make_weight_matrix(1, 200)
        self.Wv = make_weight_matrix(200, hid_dim_dec)
        self.Wg = make_weight_matrix(200, hid_dim_dec)


    def att_operation(self, conv_out, context_h, ones, batch):

        visual_feature = conv_out  # batch x 200 x h_dim
        context_feature = context_h.view(batch, self.h_dim_dec)  # batch x h_dim
        A = torch.mm(visual_feature.view(-1, self.h_dim_dec), self.Wv.permute(1, 0)).view(batch, 200, 200)  # (1 and 2 changed)
        B = torch.mm(torch.mm(context_feature, self.Wg.permute(1, 0)).view(-1, 1), ones).view(batch, 200, 200)
        AplusB = self.Tanh(A.permute(0, 2, 1) + B)  # num_ch x num_ch

        weight_batch = []
        for i in range(batch):
            weight = F.softmax(torch.mm(self.Wh, AplusB[i]).permute(1, 0), dim=0)  # (num_ch x 1) -> weights
            weight_batch.append(weight)
        att_vec_batch = torch.bmm(visual_feature.permute(0, 2, 1), torch.stack(weight_batch)).permute(2, 0, 1)

        return att_vec_batch, weight_batch

    def att_decoder(self, Ptm1, state_tuple, conv_out, At_init):

        '''
        Ptm1 : initial position (= speed vector)
        state_tuple = (h, c)
        conv_out
        At_init : dp_onehot
        '''


        # parameter setting
        batch = Ptm1.size(0)
        ones = torch.ones(size=(1, 200)).to(Ptm1)

        # initial hidden for drvsts LSTM
        zero_h = torch.zeros(self.num_layers_dec, batch, self.h_dim_dec).cuda()
        zero_c = torch.zeros(self.num_layers_dec, batch, self.h_dim_dec).cuda()
        state_tuple_drvint = (zero_h, zero_c)

        # initial input for path LSTM
        cat_pos_drvint = torch.cat((Ptm1, At_init), dim=1)
        decoder_input = self.spatial_embedding(cat_pos_drvint).view(1, batch, self.emb_dim_dec)

        weights = []
        pred_path_rel = []
        drvint_logit_seq = []
        for iter in range(self.pred_len):

            # NOTE : go through LSTM (Ptm1 & At) for Path
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            context_h, decoder_c = state_tuple

            # att operation
            att_vec_batch, weight_batch = self.att_operation(conv_out, context_h, ones, batch)
            weights.append(torch.stack(weight_batch))

            # concate hidden and attention vector
            output_concat = torch.cat((context_h, att_vec_batch), 2).view(batch, -1) # 1 x batch x 2*h_dim
            Et = self.spatial_embedding_att(output_concat) # batch x h_dim

            # predict Pt
            Pt = self.hidden2pos(Et.view(-1, self.h_dim_dec))
            pred_path_rel.append(Pt)

            # NOTE : go through LSTM (context_h, att_vect_batch, Pt) - drvint
            if (iter == 0):
                concat_h_att_pt = torch.cat((output_concat, Pt, At_init), dim=1)
            else:
                concat_h_att_pt = torch.cat((output_concat, Pt, drvint_logit_seq[iter-1]), dim=1)

            Et_drvint = self.spatial_embedding_drvint(concat_h_att_pt)
            output_drvsts, state_tuple_drvint = self.drvsts_decoder(Et_drvint.view(1, batch, self.h_dim_dec), state_tuple_drvint)
            context_h_drvsts, decoder_c_drvsts = state_tuple_drvint

            # predict Atp1
            Atp1 = self.hidden2class(context_h_drvsts.view(-1, self.h_dim_dec))
            drvint_logit_seq.append(Atp1)

            # NOTE : decoder input for the next iteration
            cat_pos_drvint = torch.cat((Pt, Atp1), dim=1)
            decoder_input = self.spatial_embedding(cat_pos_drvint).view(1, batch, self.emb_dim_dec)

        # seq_len x batch x 2, seq_len x batch x h_dim
        return torch.stack(pred_path_rel, dim=0), torch.stack(drvint_logit_seq, dim=0)[:-1, :, :], torch.stack(weights)

    def forward(self, obs_path_rel, dp_onehot, conv_out, best_k):

        # get batch size
        batch = obs_path_rel.size(1)

        # conv out embedding
        conv_out_emb = self.conv_embedding(conv_out)

        # intention embedding
        intend_vec = self.intention_embedding(dp_onehot)

        # path generation
        decoder_c = torch.zeros(self.num_layers_dec, batch, self.h_dim_dec).cuda()
        speed = torch.sqrt(torch.sum(obs_path_rel[-1] ** 2, dim=1)).repeat(2, 1).permute(1, 0)

        overall_path = []
        overall_drvsts = []
        overall_weights = []
        for z in range(best_k):

            # noise vector generation
            noise_vec = get_noise((1, self.noise_dim), 'gaussian')
            noise_vec = noise_vec.repeat(batch, 1)

            # make context vector
            context_h = torch.unsqueeze(self.mixing_mlp(torch.cat((intend_vec, noise_vec), dim=1)), 0)

            # generation, updated, 200213, intention vector and path communicate each other ?
            state_tuple = (context_h, decoder_c)
            pred_path_rel, pred_drvsts, weights = self.att_decoder(speed, state_tuple, conv_out_emb, dp_onehot)

            overall_path.append(pred_path_rel)
            overall_drvsts.append(pred_drvsts)
            overall_weights.append(weights.view(pred_path_rel.size(0), batch, 10, 20))

        return overall_path, overall_drvsts, overall_weights

class PathDiscriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 obs_len,
                 pred_len,
                 emb_dim_dis,
                 hid_dim_dis,
                 num_layers_dis,
                 onehot_dim,
                 intend_dim,
                 conv_flat_size,
                 dropout):
        super(PathDiscriminator, self).__init__()


        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.input_dim = input_dim
        self.emb_dim_dis = emb_dim_dis
        self.h_dim_dis = hid_dim_dis
        self.onehot_dim = onehot_dim
        self.intend_dim = intend_dim
        self.conv_flat_size = conv_flat_size
        self.num_layers = num_layers_dis
        self.dropout = dropout


        # conv feature embedding
        self.conv_embedding = make_mlp(dim_list=[self.conv_flat_size, self.h_dim_dis])

        # attention lstm
        self.lstm = nn.LSTM(emb_dim_dis, hid_dim_dis, num_layers_dis, dropout=dropout)
        self.spatial_embedding = make_mlp(dim_list=[input_dim, emb_dim_dis])
        self.spatial_embedding_att = make_mlp(dim_list=[emb_dim_dis + hid_dim_dis, emb_dim_dis])

        self.Tanh = nn.Tanh()
        self.Wh = make_weight_matrix(1, 200)
        self.Wv = make_weight_matrix(200, hid_dim_dis)
        self.Wg = make_weight_matrix(200, hid_dim_dis)

        # driving action lstm
        self.drvsts_embedding = make_mlp(dim_list=[onehot_dim, intend_dim])
        self.drvsts_lstm = nn.LSTM(intend_dim, intend_dim, num_layers_dis, dropout=dropout)

        # discriminator/classifier
        self.classifier = self.mlp([hid_dim_dis, onehot_dim], dropout)
        self.discriminator = self.mlp([hid_dim_dis, 1], dropout)
        self.discriminator_drvsts = self.mlp([intend_dim, 1], dropout)

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

    def att_operation(self, conv_out, context_h, ones, batch):

        visual_feature = conv_out  # batch x 200 x h_dim
        context_feature = context_h.view(batch, self.h_dim_dis)  # batch x h_dim
        A = torch.mm(visual_feature.view(-1, self.h_dim_dis), self.Wv.permute(1, 0)).view(batch, 200, 200)  # (1 and 2 changed)
        B = torch.mm(torch.mm(context_feature, self.Wg.permute(1, 0)).view(-1, 1), ones).view(batch, 200, 200)
        AplusB = self.Tanh(A.permute(0, 2, 1) + B)  # num_ch x num_ch

        weight_batch = []
        for i in range(batch):
            weight = F.softmax(torch.mm(self.Wh, AplusB[i]).permute(1, 0), dim=0)  # (num_ch x 1) -> weights
            weight_batch.append(weight)
        att_vec_batch = torch.bmm(visual_feature.permute(0, 2, 1), torch.stack(weight_batch)).permute(2, 0, 1)

        return att_vec_batch

    def att_lstm(self, path, conv_out):
        """
        Inputs:
        - traj: Tensor of shape (seq_len, batch, 2)
        - conv_out : batch size x ch x h_dim
        Output:
        - context_h
        """
        batch = path.size(1)
        seq_len = path.size(0) # [xo[-1], xp[0], ... , xp[19]

        init_h = torch.zeros(self.num_layers, batch, self.h_dim_dis).cuda()
        init_c = torch.zeros(self.num_layers, batch, self.h_dim_dis).cuda()
        state_tuple = (init_h, init_c)

        lstm_input_emb = self.spatial_embedding(path.view(-1, 2)).view(-1, batch, self.emb_dim_dis)
        ones = torch.ones(size=(1, 200)).to(path)

        att_state_list = []
        state_list = []
        for iter in range(seq_len):

            # go through LSTM
            output, state_tuple = self.lstm(lstm_input_emb[iter].view(1, batch, self.emb_dim_dis), state_tuple)

            # attention mechanism
            hidden, decoder_c = state_tuple

            # ----------------------------------------------------
            att_vec_batch = self.att_operation(conv_out, hidden, ones, batch)

            # concat hidden and attention vector
            hidden_att = self.spatial_embedding_att(torch.cat((hidden, att_vec_batch), 2))

            # stack into the lists
            att_state_list.append(hidden_att[0])
            state_list.append(hidden[0])

        att_state_stack = torch.squeeze(torch.stack(att_state_list))
        att_state_avg_pool = torch.sum(att_state_stack, 0)

        state_stack = torch.squeeze(torch.stack(state_list))
        state_avg_pool = torch.sum(state_stack, 0)

        return torch.unsqueeze(state_avg_pool, 0), torch.unsqueeze(att_state_avg_pool, 0)

    def forward(self, path_rel, conv_out, drvsts_seq):
        """
        Inputs:
        - path_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        batch = path_rel.size(1)

        # conv embedding
        conv_out_emb = self.conv_embedding(conv_out)

        # path discrimination/classification
        final_h_repr, final_h_repr_att = self.att_lstm(path_rel, conv_out_emb)  # (1 x batch x hid_dim)

        out_src = self.discriminator(final_h_repr.view(batch, self.emb_dim_dis))
        out_cls_repr = self.classifier(final_h_repr_att.view(batch, self.emb_dim_dis))


        # sequential drvints discrimination
        drvsts_seq_emb = self.drvsts_embedding(drvsts_seq.view(-1, self.onehot_dim)).view(self.pred_len-1, batch, self.intend_dim)
        context_h = torch.zeros(self.num_layers, batch, self.intend_dim).cuda()
        context_c = torch.zeros(self.num_layers, batch, self.intend_dim).cuda()
        state_tuple = (context_h, context_c)
        output, state_tuple = self.drvsts_lstm(drvsts_seq_emb, state_tuple)

        out_src_drvsts = self.discriminator_drvsts(state_tuple[0].view(batch, self.intend_dim))


        return out_src, out_cls_repr, out_src_drvsts

def Load_Overall_Models(args, dtype):

    # trajectory generator
    PathGen = PathGenerator(input_dim=args.input_dim,
                            obs_len=args.obs_seq_len,
                            pred_len=args.num_pos,
                            emb_dim_dec=args.emb_dim_dec,
                            hid_dim_dec=args.hid_dim_dec,
                            num_layers_dec=args.num_layers_dec,
                            onehot_dim=args.onehot_dim,
                            intend_dim=args.intend_dim,
                            conv_flat_size=args.conv_flat_size,
                            noise_dim=args.noise_dim,
                            dropout=args.drop_prob_gen)
    PathGen.apply(init_weights)
    PathGen.type(dtype)

    # trajectory discriminator
    PathDis = PathDiscriminator(input_dim=args.input_dim,
                                obs_len=args.obs_seq_len,
                                pred_len=args.num_pos,
                                emb_dim_dis=args.emb_dim_dis,
                                hid_dim_dis=args.hid_dim_dis,
                                num_layers_dis=args.num_layers_dis,
                                onehot_dim=args.onehot_dim,
                                intend_dim=args.intend_dim,
                                conv_flat_size=args.conv_flat_size,
                                dropout=args.drop_prob_dis)
    PathDis.apply(init_weights)
    PathDis.type(dtype)

    # conv net
    ResNet = ConvNet(use_pretrained=True,
                     feature_extract=False,
                     resent_model=args.resnet_model)
    
    if (args.multi_gpu == 1):
        ResNet = nn.DataParallel(ResNet)
    ResNet.type(dtype)

    # optimizer
    opt_g = optim.Adam(PathGen.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    opt_d = optim.Adam(PathDis.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    opt_c = optim.Adam(ResNet.parameters(), lr=args.learning_rate_cnn, betas=(args.beta1, 0.999))

    return PathGen, PathDis, ResNet, opt_g, opt_d, opt_c































