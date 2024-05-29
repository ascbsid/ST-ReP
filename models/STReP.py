import torch
import torch.nn as nn
from math import sqrt
import numpy as np

from functools import partial
from einops import rearrange, repeat
from layers.maskgenerator import MaskGenerator

class MultiHeadsAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super(MultiHeadsAttention, self).__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1, returnA=False, att_type='full'):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hid_dim//n_heads)
        d_values = d_values or (hid_dim//n_heads)

        if att_type == 'full' or att_type == 'proxy':
            self.inner_attention = MultiHeadsAttention(
                scale=None, attention_dropout=dropout, returnA=returnA)
        # elif att_type=='prob':
        #     self.inner_attention = ProbAttention(False, prob_factor, attention_dropout=dropout, output_attention=returnA)
        self.query_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.value_projection = nn.Linear(hid_dim, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hid_dim)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, A = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        if self.returnA:
            return out, A
        else:
            return out, None

# proxy is learnable parameters
class EncoderLayer(nn.Module):
    '''
    input shape: [batch_size, T, N, hid_dim]
    output shape: [batch_size, T, N, hid_dim]
    '''

    def __init__(self, factor, hid_dim, n_heads, input_length, time_factor, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1, activation='gelu',return_att=False):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att

        assert num_nodes is not None

        assert activation in ['gelu', 'relu', 'GLU','tanh']
        activation_func_dict = {'gelu': nn.GELU(),
                                'relu': nn.ReLU(),
                                'GLU': nn.GLU(),
                                'tanh':nn.Tanh()}
        # self.readout_fc = nn.Linear(num_nodes, factor)
        self.proxy_token = nn.Parameter(torch.zeros(1, factor, hid_dim))
        self.time_factor= time_factor
        self.time_readout = nn.Sequential(nn.Conv2d(input_length, time_factor, kernel_size=(1,1)), 
                                          activation_func_dict[activation],
                                          nn.Conv2d(time_factor, time_factor, kernel_size=(1,1)))
        self.time_recover = nn.Sequential(nn.Conv2d(time_factor, time_factor, kernel_size=(1,1)), 
                                          activation_func_dict[activation],
                                          nn.Conv2d(time_factor, input_length, kernel_size=(1,1)))
        
        self.node2proxy = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)
        self.proxy2node = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)

        self.dropout = nn.Dropout(dropout)

        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP2 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        batch = data.shape[0]
        T = data.shape[1]

        # now = data[:, -1, :, :]
        # temp = rearrange(now, 'b n c -> b c n')
        # z_proxy = self.readout_fc(temp)
        # z_proxy = repeat(z_proxy, 'b c k -> (b repeat) c k', repeat=self.time_factor)
        # z_proxy = rearrange(z_proxy, 'bt c K -> bt K c')

        z_proxy = repeat(self.proxy_token,'o m d -> (o b) m d', b = batch*self.time_factor)
        
        kv_data = self.time_readout(data)
        kv_data = rearrange(kv_data, 'b t n c-> (b t) n c')

        proxy_feature, A1 = self.node2proxy(z_proxy, kv_data, kv_data)
        node_feature, A2 = self.proxy2node(kv_data, proxy_feature, proxy_feature)
        enc_feature = kv_data + self.dropout(node_feature)
        enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))

        final_out = rearrange(
            enc_feature, '(b T) N hid_dim -> b T N hid_dim', b=batch)
        final_out = self.time_recover(final_out)

        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None

class simple_predictor(nn.Module):
    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 activation='gelu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.hid_dim = hid_dim
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        self.end_conv1 = nn.Conv2d(in_channels=input_length, out_channels=predict_length, kernel_size=1, bias=True)
        self.end_conv2 = nn.Linear(hid_dim, pre_dim)

    def forward(self, data):
        rep = self.end_conv1(data)
        data = self.end_conv2(rep)
        return data



class get_SpatialEmb(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, hid_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, spatial_indexs=None):
        if spatial_indexs is None:
            batch, _,  num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(
                torch.arange(num_nodes))  # (N,)
        spatial_emb = self.spatial_embedding(
            spatial_indexs.to(x.device)).unsqueeze(0).unsqueeze(1) # (N, d)->(1, 1, N, d)
        return spatial_emb


class get_TemporalEmb(nn.Module):
    def __init__(self, in_dim, hid_dim, slice_size_per_day, dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, hid_dim)
        self.day_in_week_embedding = nn.Embedding(7, hid_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, t_hour, t_day, t_idx=None):
        time_in_day_emb = self.time_in_day_embedding(t_hour) #BT1d
        day_in_week_emb = self.day_in_week_embedding(t_day)         
        return time_in_day_emb, day_in_week_emb



class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, hasCross=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        in_units = in_dim*2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, latestX):
        if self.hasCross:
            t_len = x.shape[1]
            if latestX.shape[1]!= t_len :
                latestX=latestX.repeat([1, t_len, 1, 1])
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data

        

class Reconstructor(nn.Module):
    def __init__(self, num_nodes, input_length,  hid_dim, in_dim, activation='gelu', hasRes=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.hid_dim = hid_dim
        self.hasRes = hasRes
        assert activation in ['gelu', 'relu']
        self.end_conv1 = nn.Sequential(nn.Conv2d(in_channels=input_length, out_channels=input_length, kernel_size=1, bias=True),
                                       nn.GELU() if activation == 'gelu' else nn.ReLU())
        self.end_conv2 = nn.Linear(hid_dim, in_dim)


    def forward(self, data, y_time_mark = None):
        skip = data
        rep = self.end_conv1(data)
        if self.hasRes: rep = rep + skip
        data = self.end_conv2(rep)
        return data

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.in_dim if args.pre_dim is None else args.pre_dim
        self.num_nodes = args.num_nodes
        self.tau = args.tau
        self.useTCN = args.tau > 0
        self.hid_dim = args.hid_dim
        self.hasCross = bool(args.hasCross)
        self.num_layers = args.num_layers
        self.pred_hasRes = bool(args.pred_hasRes)
        self.mask_ratio = args.mask_ratio
        self.mask_only = args.loss_type == 'mask_only'
        self.pred_only = args.loss_type == 'pred_only'

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, args.hid_dim))
        self.get_spatial_emb = get_SpatialEmb(args.in_dim, args.hid_dim, args.num_nodes, args.st_emb_dropout)
        self.get_temporal_emb = get_TemporalEmb(args.in_dim, args.hid_dim, args.slice_size_per_day, args.st_emb_dropout)
        self.data_encoding = DataEncoding(args.in_dim, args.hid_dim, self.hasCross, activation=args.activation_data)

        if self.useTCN:
            if args.tau == 2:
                tcn_pad_l, tcn_pad_r = 0, 1
                self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
            elif args.tau == 3:
                tcn_pad_l, tcn_pad_r = 1, 1
                self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
            self.time_conv = nn.Conv2d(args.hid_dim, args.hid_dim, (1, args.tau))

        self.spatial_agg_list = nn.ModuleList([
                    EncoderLayer(args.M, args.hid_dim, n_heads=args.n_heads, num_nodes=args.num_nodes, dropout=args.spatial_dropout,
                                att_dropout=args.spatial_att_dropout, activation=args.activation_enc,
                                time_factor=args.time_factor, input_length=args.input_length) for _ in range(args.num_layers)])
        self.rep_dropout = nn.Dropout(args.rep_dropout)
        self.predictor = simple_predictor(
                args.num_nodes, args.input_length, args.predict_length, args.hid_dim, pre_dim=self.pre_dim, activation=args.activation_dec)
        self.reconstructor = Reconstructor(args.num_nodes, args.input_length, args.hid_dim, args.in_dim, activation=args.activation_dec, hasRes=self.pred_hasRes)
      
        self.note = f'{args.input_length}to{args.predict_length}_{args.ds_input_length}to{args.ds_predict_length}_{args.note}'
    
    def _getnote(self):
        return self.note
    
    def _input_processing(self, x):
        inputx = x[0]
        if len(x)==2:
            x_data, x_time_mark = x  # x: (B,T,N,C) t_hour:(B,T,1) t_day:(B,T,1) t_idx:(B,T,1)
            B, T, N, _ = x_data.shape
        elif len(x)==3:
            x_data, x_time_mark, y_time_mark = x
            B, T, N, _ = x_data.shape
            y_t_hour = y_time_mark[...,0:1]
            y_t_day = y_time_mark[...,1:2] 
        elif len(x)==4:
            x_data, y_data, x_time_mark, y_time_mark = x
            B, T, N, _ = x_data.shape
            y_t_hour = y_time_mark[...,0:1]
            y_t_day = y_time_mark[...,1:2] 

        
        x_t_hour = x_time_mark[...,0:1]
        x_t_day = x_time_mark[...,1:2] 
        latestX = x_data[:, -1:, :, :]

        if len(x) == 4:
            return inputx, latestX, x_t_hour, x_t_day, y_data, y_t_hour, y_t_day
        else:
            return inputx, latestX, x_t_hour, x_t_day
    
    def _mask_raw_data(self, data, unmasked_token_index=None, masked_token_index=None):
        B, T, N, C = data.shape

        MaskEngine = MaskGenerator(T, self.mask_ratio)
        unmasked_token_index, masked_token_index = MaskEngine.uniform_rand()
        data_o = data[:, unmasked_token_index, :, :]   

        return data_o, unmasked_token_index, masked_token_index

    def get_rep(self, x):
        group =self._input_processing(x)
        if len(x) == 4:
            inputx, latestX, x_t_hour, x_t_day, inputy, y_t_hour, y_t_day = group
        else:
            inputx, latestX, x_t_hour, x_t_day = group

        spatial_emb = self.get_spatial_emb(inputx)
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)
        data = self.data_encoding(inputx, latestX)
        data = data + x_tod_emb + x_dow_emb+ spatial_emb
        T = inputx.shape[1]
        if self.useTCN:
            data = data.transpose(1, 3)  # (B,T,N,C)->(B,C,N,T)
            if self.tau > 1:
                data = self.padding(data)
            data = self.time_conv(data)
            data = data.transpose(1, 3)  # (B,C,N,T)->(B,T,N,C)
            assert data.shape[1] == T
        rep = self._encode(data)
        return rep, latestX

    def _encode(self, data):
        skip = data
        A_list = []
        for i in range(self.num_layers):
            data, _ = self.spatial_agg_list[i](data)
        data += skip

        return data

    
    def forward(self, x):
        group = self._input_processing(x)
        assert len(x) == 4
        inputx, latestX, x_t_hour, x_t_day, inputy, y_t_hour, y_t_day = group
        

        spatial_emb = self.get_spatial_emb(inputx)
        x_tod_emb, x_dow_emb = self.get_temporal_emb(x_t_hour, x_t_day)
        x_o, x_unmasked_token_index, x_masked_token_index =self._mask_raw_data(inputx)
        

        x_tod_emb_o = x_tod_emb[:, x_unmasked_token_index, :, :]
        x_dow_emb_o = x_dow_emb[:, x_unmasked_token_index, :, :]
        semb_o = spatial_emb

        x_data_o  = self.data_encoding(x_o, latestX)
        x_data_o =  x_data_o + x_tod_emb_o + x_dow_emb_o+ semb_o
        T_o = x_data_o.shape[1]

        if self.useTCN:
            x_data_o = x_data_o.transpose(1, 3)  # (B,T,N,C)->(B,C,N,T)
            if self.tau > 1:
                x_data_o = self.padding(x_data_o)
            x_data_o = self.time_conv(x_data_o)
            x_data_o = x_data_o.transpose(1, 3)  # (B,C,N,T)->(B,T,N,C)
            assert x_data_o.shape[1] == T_o

        B, T_, N_, d = x_data_o.shape
        x_tod_emb_m = x_tod_emb[:, x_masked_token_index,:,:] # B, T, 1, d
        x_dow_emb_m = x_dow_emb[:, x_masked_token_index,:,:]
        x_mask_token = self.mask_token.expand(B, len(x_masked_token_index), N_, d)
        x_mask_token = x_mask_token + x_tod_emb_m + x_dow_emb_m
        x_data_full = torch.cat([x_mask_token, x_data_o], dim=1)

        x_rep_full = self._encode(x_data_full)

        if self.pred_only:
            reconx = None
        else:
            reconx = self.reconstructor(x_rep_full)


        if self.mask_only:
            predy = None
        else:
            x_rep_drop = self.rep_dropout(x_rep_full)
            predy = self.predictor(x_rep_drop)

        return inputx, reconx, inputy, predy

if __name__ == "__main__":
    x = torch.randn((2, 12, 307, 1))
    t_hour = torch.LongTensor(torch.randint(0, 287, (2, 12, 1)))
    t_day = torch.LongTensor(torch.randint(0, 6, (2, 12, 1)))
    layer = Model(12, 1, 307, 1, att_type='proxy')
    y, A = layer([x, t_hour, t_day])
    print(layer)
    print(y.shape)
    print(type(A))
