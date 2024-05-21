# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def shredFacts(facts,hiss,ent_hiss,raw_facts=None):
    heads = torch.tensor(facts[:, 0]).long().cuda()
    rels = torch.tensor(facts[:, 1]).long().cuda()
    tails = torch.tensor(facts[:, 2]).long().cuda()
    dates = torch.tensor(raw_facts[:, 3]).long().cuda()
    dateid=torch.tensor(facts[:, 3]).long().cuda()
    # numpy.array()
    ent_hiss =torch.tensor(ent_hiss).long().cuda()
    hiss =torch.tensor(hiss).long().cuda()
    return heads, rels, tails, dates,hiss,ent_hiss,dateid
