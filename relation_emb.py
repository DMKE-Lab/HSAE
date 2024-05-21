# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from params import Params
from transformer import Encoder


class Rel_time_emb(torch.nn.Module):
    def __init__(self, dataset, params):
        super(Rel_time_emb, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.tim_embs_f = nn.Embedding(dataset.numTime(),params.emb_dim).cuda()

        self.h_map_emb = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.t_map_emb = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.rel_emb_h = nn.Embedding(dataset.numRel()+1, params.emb_dim).cuda()
        self.rel_emb_t = nn.Embedding(dataset.numRel()+1, params.emb_dim).cuda()
        self.rel_emb_q = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.tim_embs_f.weight)

        nn.init.xavier_uniform_(self.h_map_emb.weight)
        nn.init.xavier_uniform_(self.t_map_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb_h.weight)
        nn.init.xavier_uniform_(self.rel_emb_t.weight)
        nn.init.xavier_uniform_(self.rel_emb_q.weight)
        
        self.his_encoder = Encoder(self.params)

    def getRelEmbeddings(self,heads, rels, tails, hiss, timeid,ent_type,train_or_test):

        if ent_type == "subs":
            
            if train_or_test == "train":
                ents_for_rel, rels_his = heads.view(-1, (1 + self.params.neg_ratio))[:, 0], rels.view(-1, (1 + self.params.neg_ratio))[:, 0]
                ents_emb, rels_emb, rel_set_emb = self.h_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_h(hiss)
                pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb,timeid, hiss)
                pre_rel_emb = pre_rel_emb.unsqueeze(1)
                pre_rel_emb = pre_rel_emb.repeat(1, (1 + self.params.neg_ratio), 1).contiguous().view(-1, self.params.emb_dim)
            else:
                ents_for_rel, rels_his = heads.view(-1,1)[0], rels.view(-1,1)[0]
                if hiss.size(1) != 0:
                    ents_emb, rels_emb, rel_set_emb = self.h_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_h(hiss)
                    pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, timeid, hiss)
                    pre_rel_emb = pre_rel_emb.repeat(rels.size(0), 1).contiguous().view(-1, self.params.emb_dim)
                else:
                    pre_rel_emb = torch.zeros([rels.size(0), self.params.emb_dim]).float().cuda()
        else:
            
            if train_or_test == "train": 
                ents_for_rel, rels_his = tails.view(-1, (1 + self.params.neg_ratio))[:,0], rels.view(-1, (1 + self.params.neg_ratio))[:,0]
                ents_emb, rels_emb, rel_set_emb = self.t_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_t(hiss)
                pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, timeid, hiss)
                pre_rel_emb = pre_rel_emb.unsqueeze(1)
                pre_rel_emb = pre_rel_emb.repeat(1, (1 + self.params.neg_ratio), 1).contiguous().view(-1, self.params.emb_dim)
            else:
                ents_for_rel, rels_his = tails.view(-1,1)[0], rels.view(-1,1)[0]
                if hiss.size(1) != 0:
                    ents_emb, rels_emb, rel_set_emb = self.t_map_emb(ents_for_rel), self.rel_emb_q(rels_his), self.rel_emb_t(hiss)
                    pre_rel_emb = self.get_rel_embedds(ents_emb, rels_emb, rel_set_emb, timeid, hiss)
                    pre_rel_emb = pre_rel_emb.repeat(rels.size(0), 1).contiguous().view(-1, self.params.emb_dim)
                else:
                    pre_rel_emb = torch.zeros([rels.size(0), self.params.emb_dim]).float().cuda()
        return pre_rel_emb

    def get_rel_embedds(self, ents_emb, rels_emb, rel_set_emb, timeid, historys):
        timeid = timeid.view(-1,1)
        pre_rel_emb, pre_attn = self.get_pre_embedd(ents_emb, rels_emb, rel_set_emb, timeid, historys)
        return pre_rel_emb

    def get_pre_embedd(self, ents_emb, rels_emb, rel_set_emb,timeid, historys):
        times_emb = self.tim_embs_f(timeid)
        # mask_attn torch.Size([256, 1, 78])
        mask_attn = self.get_mask(historys)
        pre_his_emb, pre_rels = self.make_his_emb(ents_emb, times_emb, rels_emb, rel_set_emb)
        pre_rel_emb, pre_attn = self.his_encoder(pre_rels, pre_his_emb, mask_attn)
        return pre_rel_emb, pre_attn

    def get_mask(self, his):
        assert isinstance(his, torch.Tensor)
        idx, len_q = self.dataset.numRel(), 1
        batch_size, len_k = his.size()
        mask_attn = his.data.eq(idx).unsqueeze(1)
        return mask_attn.expand(batch_size, len_q, len_k)
    
    def make_his_emb(self, ents, times, rels_emb, rel_set_emb):
        # ents  torch.Size([256, 100])
        # times   torch.Size([256, 1, 100])
        # rels_emb  torch.Size([256, 100])
        # rel_set_emb  torch.Size([256, 78, 100])
        batch_size, size, dim = rel_set_emb.size()
        rels_emb = rels_emb.unsqueeze(1)
        # rels_emb  torch.Size([256, 1, 100])
        ents = ents.unsqueeze(1)
        # ents  torch.Size([256, 1, 100])
        

        ents_new = ents.expand(batch_size, size, dim)
        # ents_new  torch.Size([256, 78, 100])
        
        times_new = times.expand(batch_size, size, dim)
        # torch.Size([256, 78, 100])
        
        his_set_emb = torch.sum(rel_set_emb * ents_new, dim=2, keepdim=True) * times_new
        
        que_rels = torch.sum(rels_emb * ents, dim=2, keepdim=True) * times
        return his_set_emb, que_rels.view(batch_size, -1, self.params.emb_dim)
        
