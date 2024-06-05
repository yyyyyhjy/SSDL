import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from loss import contrastive_loss, compute_mi
from mutual_info import logsumexp, log_density, log_importance_weight_matrix
import math
from WGAT import WGATConv
from HetEmb import HetEmb

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)

class SelfMove(nn.Module):
    def __init__(self, emb_dim, out_size,num_heads,dropout,node_num,s_dim,r_dim,hidden_units):
        super(SelfMove, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.node_embs = nn.Embedding(num_embeddings=node_num, embedding_dim=emb_dim)
        self.graph_layer = WGATConv(in_channels=emb_dim, out_channels=out_size, heads=num_heads,
                                         dropout=dropout)
        self.ct_dim=emb_dim
        self.hetEmb = HetEmb(ct_dim=self.ct_dim, time_num=48, cat_num=201)

        self.s_dim = s_dim  # time-invariant
        self.r_dim = r_dim  # time-varying
        self.e_dim = emb_dim

        self.hidden_dim = hidden_units # opt.rnn_size

        self.W=nn.Linear(emb_dim+self.ct_dim*2,self.e_dim)

        self.r_prior_gru_ly1 = nn.GRUCell(self.r_dim, self.hidden_dim)
        self.r_prior_gru_ly2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.r_prior_mean = nn.Linear(self.hidden_dim, self.r_dim)
        self.r_prior_logvar = nn.Linear(self.hidden_dim, self.r_dim)

        self.s_mean = LinearUnit(self.hidden_dim*2, self.s_dim, False)
        self.s_logvar = LinearUnit(self.hidden_dim*2, self.s_dim, False)
        self.r_mean = nn.Linear(self.hidden_dim, self.r_dim)
        self.r_logvar = nn.Linear(self.hidden_dim, self.r_dim)

        self.bi_rnn = nn.GRU(self.e_dim, self.hidden_dim, batch_first=True,dropout=0.5,bidirectional=True)
        self.uni_rnn = nn.GRU(self.hidden_dim*2, self.hidden_dim, batch_first=True,dropout=0.5)

        # history
        self.mlp = nn.Linear(emb_dim+self.ct_dim*2, 128)
        # attention layer
        self.hidden_size = hidden_units
        self.attn_size = hidden_units
        self.query = nn.Linear(128, self.hidden_size, bias=True)
        self.attn = nn.Linear(self.hidden_size, self.attn_size, bias=True)
        self.v = nn.Linear(self.hidden_size, 1)

        #self.fc_ht=nn.Linear(self.s_dim + self.r_dim,self.hidden_size)
        self.fc_zt=nn.Linear(self.r_dim,self.hidden_size)

        #MLP
        self.output = nn.Linear(self.hidden_dim*3, node_num)
        self.dropout=nn.Dropout(0.5)

    def get_history_layer(self, x, ht):
        x = self.query(x)

        v = torch.tanh(torch.mul(ht, self.attn(x)))
        vu = self.v(v)
        alphas = F.softmax(vu, dim=1)
        output = torch.mul(x, alphas)
        output = torch.sum(output, dim=1)
        output=self.dropout(output)
        return output

    def encode_and_sample_post(self, x):
        L = x.shape[1]
        # pass to one direction rnn
        features, state = self.bi_rnn(x)
        backward = features[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = features[:, L - 1, 0:self.hidden_dim]
        gru_out_s = torch.cat((frontal, backward), dim=1)
        s_mean = self.s_mean(gru_out_s)
        s_logvar = self.s_logvar(gru_out_s)
        s_post = self.reparameterize(s_mean, s_logvar, random_sampling=True)

        r_features, r_state = self.uni_rnn(features)
        r_mean = self.r_mean(r_features)
        r_logvar = self.r_logvar(r_features)
        r_post = self.reparameterize(r_mean, r_logvar, random_sampling=True)

        return s_mean, s_logvar, s_post, r_mean, r_logvar, r_post, gru_out_s


    def cal_loss(self, s_mean, s_logvar, s_post, r_post_mean, r_post_logvar, r_post, r_prior_mean, r_prior_logvar,
                 r_prior, predict_x, label):
        batch_size, n_frame, z_dim = r_post_mean.size()
        mi_xs = compute_mi(s_post, (s_mean, s_logvar))
        n_bs = r_post.shape[0]

        mi_xzs = [compute_mi(r_post_t, (r_post_mean_t, r_post_logvar_t)) \
                  for r_post_t, r_post_mean_t, r_post_logvar_t in \
                  zip(r_post.permute(1, 0, 2), r_post_mean.permute(1, 0, 2), r_post_logvar.permute(1, 0, 2))]
        mi_xz = torch.stack(mi_xzs).sum()

        l_predict = nn.CrossEntropyLoss()
        l_predict = l_predict(predict_x, label)

        s_mean = s_mean.view((-1, s_mean.shape[-1]))
        s_logvar = s_logvar.view((-1, s_logvar.shape[-1]))
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))

        r_post_var = torch.exp(r_post_logvar)
        r_prior_var = torch.exp(r_prior_logvar)
        kld_r = 0.5 * torch.sum(r_prior_logvar - r_post_logvar +
                                ((r_post_var + torch.pow(r_post_mean - r_prior_mean, 2)) / r_prior_var) - 1)
        kld_s, kld_r = kld_s / batch_size, kld_r / batch_size
        # calculate the mutual infomation
        mi_sr = torch.zeros((1)).cuda()
        if True:  # 0.1
            _logq_s_tmp = log_density(
                s_post.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, self.s_dim),
                s_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, self.s_dim),
                s_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size,self.s_dim))
            _logq_r_tmp = log_density(r_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim),
                                      r_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim),
                                      r_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size,z_dim))

            _logq_sr_tmp = torch.cat((_logq_s_tmp, _logq_r_tmp), dim=3)

            logq_s = (logsumexp(_logq_s_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255))
            logq_r = (logsumexp(_logq_r_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255))
            logq_sr = (logsumexp(_logq_sr_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255))
            mi_sr = F.relu(logq_sr - logq_s - logq_r).mean()

        loss = l_predict + 0.1 * (kld_s + kld_r + mi_sr)
        return loss

    def forward(self, graphs,batch_x,label,history_x,batch_time,history_time,batch_cat,history_cat,poi_time,poi_cat,pos_encoding):
        L = batch_x.shape[1]
        h_length = history_x.shape[1]

        # random embedding
        p_h = self.node_embs(graphs.x.to(self.device)).to(self.device)
        # graph embedding
        p_h = self.graph_layer(p_h, graphs.edge_index.to(self.device), graphs.edge_attr.to(self.device))
        embedding = nn.Embedding.from_pretrained(p_h)
        p_h = embedding(batch_x)
        history_h = embedding(history_x)

        pt_emb, pc_emb, time_emb, cat_emb = self.hetEmb(poi_time, poi_cat)
        pt_emb = nn.Embedding.from_pretrained(pt_emb)
        pc_emb = nn.Embedding.from_pretrained(pc_emb)

        time_emb = nn.Embedding.from_pretrained(time_emb)
        time_embs = time_emb(batch_time)
        history_time_embs = time_emb(history_time)
        cat_emb = nn.Embedding.from_pretrained(cat_emb)
        cat_embs = cat_emb(batch_cat)
        history_cat_embs = cat_emb(history_cat)

        # poi-time-cat embedding concat
        p_h = torch.cat((p_h, time_embs, cat_embs), dim=2)
        p_h=self.W(p_h)

        s_mean, s_logvar, s_post, r_mean_post, r_logvar_post, r_post ,out= self.encode_and_sample_post(p_h)
        r_mean_prior, r_logvar_prior, r_prior = self.sample_r_prior_train(r_post,  random_sampling=self.training)

        z_flatten = r_post.view(-1, r_post.shape[2])
        s_expand = s_post.unsqueeze(1).expand(-1, L, self.s_dim)
        zf = torch.cat((r_post, s_expand), dim=2)  # (B,L,f_dim+z_dim)
        #ht=zf[:,-1,:].unsqueeze(1)
        ht = r_post[:, -1, :].unsqueeze(1)
        softplus=nn.Softplus()
        ht=softplus(self.fc_zt(ht))
        #ht=F.relu(self.fc_ht(ht))

        history_h = torch.cat((history_h, history_time_embs, history_cat_embs), dim=2)
        history_h = self.mlp(history_h)
        history_h *= math.sqrt(128)
        history_h += pos_encoding[:, :h_length, :]

        history_out = self.get_history_layer(history_h, ht)

        out = torch.cat((out, history_out), dim=-1)
        predict_x = self.output(out).squeeze(1)
        loss = self.cal_loss(s_mean, s_logvar, s_post, r_mean_post, r_logvar_post, r_post, r_mean_prior, r_logvar_prior,r_prior, predict_x, label)
        l_predict = nn.CrossEntropyLoss()
        l_predict = l_predict(predict_x, label)

        loss+=l_predict
        predict_x = F.softmax(predict_x)

        return predict_x,loss

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_r_prior_train(self, r_post, random_sampling=True):
        r_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        r_means = None
        r_logvars = None
        batch_size, L = r_post.shape[0], r_post.shape[1]

        r_t = torch.zeros(batch_size, self.r_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(L):
            # two layer LSTM and two one-layer FC
            h_t_ly1 = self.r_prior_gru_ly1(r_t, h_t_ly1)
            h_t_ly2 = self.r_prior_gru_ly2(h_t_ly1, h_t_ly2)

            r_mean_t = self.r_prior_mean(h_t_ly2)
            r_logvar_t = self.r_prior_logvar(h_t_ly2)
            r_prior = self.reparameterize(r_mean_t, r_logvar_t, random_sampling)
            if r_out is None:
                r_out = r_prior.unsqueeze(1)
                r_means = r_mean_t.unsqueeze(1)
                r_logvars = r_logvar_t.unsqueeze(1)
            else:
                r_out = torch.cat((r_out, r_prior.unsqueeze(1)), dim=1)
                r_means = torch.cat((r_means, r_mean_t.unsqueeze(1)), dim=1)
                r_logvars = torch.cat((r_logvars, r_logvar_t.unsqueeze(1)), dim=1)
            r_t = r_post[:, i, :]
        return r_means, r_logvars, r_out






