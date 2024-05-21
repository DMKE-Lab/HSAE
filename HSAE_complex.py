# -*- coding: utf-8 -*-

# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from entity_emb import Ent_time_emb
from relation_emb import Rel_time_emb


class HSAE_complex(torch.nn.Module):
    def __init__(self, dataset, params):
        super(HSAE_complex, self).__init__()
        self.dataset = dataset
        self.params = params

        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        self.tim_embs_f = nn.Embedding(dataset.numTime(),params.t_emb_dim).cuda()

        self.time_nl = torch.sin
        self.Ent_emb=Ent_time_emb(self.dataset,self.params)
        self.Rel_emb=Rel_time_emb(self.dataset,self.params)

        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
        nn.init.xavier_uniform_(self.tim_embs_f.weight)
        


    def getEmbeddings(self, batch,ent_type,train_or_test):
        
        heads, rels, tails,dates,hiss,ent_hiss,dateid=batch
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        T_embs1 = self.tim_embs_f(dateid)
        T_embs2 = torch.ones(dateid.__len__(), self.params.s_emb_dim).cuda()
        T_embs1 = torch.cat((T_embs1, T_embs2), 1)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        pre_ent_emb1=self.Ent_emb.getRelEmbeddings(heads, rels, tails,ent_hiss, dates,"subs",train_or_test)
        pre_ent_emb2=self.Ent_emb.getRelEmbeddings(heads, rels, tails,ent_hiss, dates,"objs",train_or_test)
        
        h_embs1 = (1-self.params.alp)*h_embs1 + self.params.alp*pre_ent_emb1
        t_embs1 = (1-self.params.alp)*t_embs1 + self.params.alp*pre_ent_emb1

        h_embs2 = (1-self.params.alp)*h_embs2 + self.params.alp*pre_ent_emb2
        t_embs2 = (1-self.params.alp)*t_embs2 + self.params.alp*pre_ent_emb2
               
        pre_rel_emb=self.Rel_emb.getRelEmbeddings(heads, rels, tails,hiss, dates,ent_type,train_or_test)
        
        r_embs1 = (1-self.params.alp)*r_embs1 + self.params.alp*pre_rel_emb
        r_embs2 = (1-self.params.alp)*r_embs2 + self.params.alp*pre_rel_emb

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2,T_embs1

    def forward(self, batch1, batch2=None, train_or_test="train", ent_type="subs"):
        if batch2 == None:
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 ,T_embs1 = self.getEmbeddings(batch1, ent_type, train_or_test)
            scores = h_embs1*t_embs1*r_embs1*T_embs1+h_embs2*t_embs2*r_embs1*T_embs1+h_embs1*t_embs2*r_embs2*T_embs1-h_embs2*t_embs1*r_embs2*T_embs1
        else:
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 ,T_embs1 = self.getEmbeddings(batch1, "objs", train_or_test)
            scores1 = h_embs1*t_embs1*r_embs1*T_embs1+h_embs2*t_embs2*r_embs1*T_embs1+h_embs1*t_embs2*r_embs2*T_embs1-h_embs2*t_embs1*r_embs2*T_embs1

            h_embs3, r_embs3, t_embs3, h_embs4, r_embs4, t_embs4 ,T_embs2 = self.getEmbeddings(batch2, "subs", train_or_test)
            scores2 = h_embs3*t_embs3*r_embs3*T_embs2+h_embs4*t_embs4*r_embs3*T_embs2+h_embs3*t_embs4*r_embs4*T_embs2-h_embs4*t_embs3*r_embs4*T_embs2

            scores = torch.cat((scores1, scores2), 0)
            
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
      
