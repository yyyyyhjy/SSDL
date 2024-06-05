import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class HetEmb(nn.Module):
    def __init__(self, ct_dim,time_num,cat_num):
        super(HetEmb, self).__init__()
        ##
        self.fc1 = nn.Linear(time_num, ct_dim)
        self.fc2 = nn.Linear(cat_num,ct_dim)
    """
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(mean = 0, std = 0.1)
                m.bias.data.normal_(mean = 0, std = 0.1)
                """
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self,poi_time,poi_cat):
        pt_emb=self.fc1(poi_time)
        # pt_emb=F.relu(pt_emb)
        pc_emb=self.fc2(poi_cat)
        # pc_emb=F.relu(pc_emb)
        time_emb=self.fc1.weight.permute(1,0)
        cat_emb=self.fc2.weight.permute(1,0)

        return pt_emb,pc_emb,time_emb,cat_emb