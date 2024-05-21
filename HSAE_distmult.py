# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from entity_emb import Ent_time_emb
from relation_emb import Rel_time_emb


class HSAE_distmult(torch.nn.Module):
    def __init__(self, dataset, params):
        super(HSAE_distmult, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.rel_embs = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        self.tim_embs = nn.Embedding(dataset.numTime(), params.t_emb_dim).cuda()
    
        self.Ent_emb=Ent_time_emb(self.dataset,self.params)
        self.Rel_emb=Rel_time_emb(self.dataset,self.params)
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.tim_embs.weight)

    def getEmbeddings(self, batch,ent_type,train_or_test):

        heads, rels, tails,dates,hiss,ent_hiss,dateid=batch
        h_embs1 = self.ent_embs(heads)
        r_embs1 = self.rel_embs(rels)
        t_embs1 = self.ent_embs(tails)
        T_embs1 = self.tim_embs(dateid)
        T_embs2 = torch.ones(dateid.__len__(), self.params.s_emb_dim).cuda()
        T_embs1 = torch.cat((T_embs1, T_embs2), 1)
       
        pre_ent_emb=self.Ent_emb.getRelEmbeddings(heads, rels, tails,ent_hiss, dates,ent_type,train_or_test)
        
        
        h_embs1 = (1-self.params.alp)*h_embs1 + self.params.alp*pre_ent_emb
        t_embs1 = (1-self.params.alp)*t_embs1 + self.params.alp*pre_ent_emb

        pre_rel_emb = self.Rel_emb.getRelEmbeddings(heads, rels, tails,hiss, dates,ent_type,train_or_test)

        r_embs1 = (1-self.params.alp)*r_embs1 + self.params.alp*pre_rel_emb

        return h_embs1, r_embs1, t_embs1, T_embs1

    def forward(self, batch1=None, batch2=None, train_or_test="train", ent_type="subs"):
        if batch2 == None:
            h_embs, r_embs, t_embs,T_embs1 = self.getEmbeddings(batch1, ent_type, train_or_test)
            scores = h_embs* r_embs *T_embs1* t_embs

        else:
            h_embs, r_embs, t_embs,T_embs1 = self.getEmbeddings(batch1, "objs", train_or_test)
            scores1 = h_embs* r_embs*T_embs1* t_embs

            h_embs2, r_embs2, t_embs2,T_embs2 = self.getEmbeddings(batch2, "subs", train_or_test)
            scores2 = h_embs2* r_embs2 *T_embs2* t_embs2
            
            scores = torch.cat((scores1, scores2), 0)
        
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores
        