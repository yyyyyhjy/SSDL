import torch
import torch.nn as nn
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

class PreSelfMove(nn.Module):
    def __init__(self, emb_dim, out_size,num_heads,dropout,node_num,s_dim,r_dim,hidden_units):
        super(PreSelfMove, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.s_dim = s_dim  # time-invariant
        self.r_dim = r_dim # time-varying
        self.e_dim = emb_dim

        self.node_embs = nn.Embedding(num_embeddings=node_num, embedding_dim=emb_dim)
        self.graph_layer = WGATConv(in_channels=emb_dim, out_channels=out_size, heads=num_heads,
                                         dropout=dropout)
        self.ct_dim=emb_dim
        self.hetEmb = HetEmb(ct_dim=self.ct_dim, time_num=48, cat_num=201)

        self.hidden_dim = hidden_units  # opt.rnn_size
        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.r_prior_gru_ly1 = nn.GRUCell(self.r_dim, self.hidden_dim)
        self.r_prior_gru_ly2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.r_prior_mean = nn.Linear(self.hidden_dim, self.r_dim)
        self.r_prior_logvar = nn.Linear(self.hidden_dim, self.r_dim)

        self.W = nn.Linear(emb_dim + self.ct_dim * 2, self.e_dim)
        self.bi_rnn = nn.GRU(self.e_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.uni_rnn = nn.GRU(self.hidden_dim * 2, self.hidden_dim, batch_first=True)

        self.s_mean = LinearUnit(self.hidden_dim*2, self.s_dim, False)
        self.s_logvar = LinearUnit(self.hidden_dim*2, self.s_dim, False)
        self.r_mean = nn.Linear(self.hidden_dim, self.r_dim)
        self.r_logvar = nn.Linear(self.hidden_dim, self.r_dim)

        self.decoder = nn.Linear(self.s_dim + self.r_dim, node_num)
        self.dropout=nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.constant_(m.bias_ih_l0, 0)
                nn.init.constant_(m.bias_hh_l0, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                nn.init.normal_(m.bias.data, mean=0, std=0.1)

    def encode_and_sample_post(self, x):
        B,L=x.shape[0],x.shape[1]

        features, state = self.bi_rnn(x)
        backward = features[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = features[:, L - 1, 0:self.hidden_dim]
        gru_out_s = torch.cat((frontal, backward), dim=1)
        s_mean = self.s_mean(gru_out_s)
        s_logvar = self.s_logvar(gru_out_s)
        s_post = self.reparameterize(s_mean, s_logvar, random_sampling=True)

        r_features,r_state=self.uni_rnn(features)
        r_mean = self.r_mean(r_features)
        r_logvar = self.r_logvar(r_features)
        r_post = self.reparameterize(r_mean, r_logvar, random_sampling=True)

        return s_mean, s_logvar, s_post, r_mean, r_logvar, r_post


    def forward(self, graphs,batch_x,batch_time,batch_cat,poi_time,poi_cat):
        L=batch_x.shape[1]

        # random embedding
        p_h = self.node_embs(graphs.x.to(self.device)).to(self.device)
        # graph
        p_h = self.graph_layer(p_h, graphs.edge_index.to(self.device), graphs.edge_attr.to(self.device))
        embedding = nn.Embedding.from_pretrained(p_h)
        p_h = embedding(batch_x)

        pt_emb, pc_emb, time_emb, cat_emb = self.hetEmb(poi_time, poi_cat)
        time_emb = nn.Embedding.from_pretrained(time_emb)
        time_embs = time_emb(batch_time)
        cat_emb = nn.Embedding.from_pretrained(cat_emb)
        cat_embs = cat_emb(batch_cat)

        # poi-time-cat embedding concat
        p_h = torch.cat((p_h, time_embs, cat_embs), dim=2)
        p_h=self.W(p_h)

        s_mean, s_logvar, s_post, r_mean_post, r_logvar_post, r_post = self.encode_and_sample_post(p_h)
        r_mean_prior, r_logvar_prior, r_prior = self.sample_r_prior_train(r_post, random_sampling=self.training)

        r_flatten = r_post.view(-1, r_post.shape[2])
        s_expand = s_post.unsqueeze(1).expand(-1, L, self.s_dim)
        zf = torch.cat((r_post, s_expand), dim=2)

        recon_x = self.decoder(zf)
        return s_mean, s_logvar, s_post, r_mean_post, r_logvar_post, r_post, r_mean_prior, r_logvar_prior, r_prior, recon_x


    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_r_prior_train(self, r_post, random_sampling=True):
        r_out = None
        r_means = None
        r_logvars = None
        batch_size,L = r_post.shape[0], r_post.shape[1]

        r_t = torch.zeros(batch_size, self.r_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(L):
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
            r_t = r_post[:,i,:]
        return r_means, r_logvars, r_out