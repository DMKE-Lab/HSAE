# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch

from measure import Measure
from scripts import shredFacts


class Tester:
    def __init__(self, dataset, model_path, valid_or_test,model_name):
        self.model = torch.load(model_path)
        self.model.eval()
        self.model_name=model_name
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        
    def getRank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1
    
    def replaceAndShred(self,fact,hiss,ent_hiss,raw_or_fil,head_or_tail):
        head,rel,tail,date=fact
        if head_or_tail == "head":
            ret_facts = [(i, rel, tail, date) for i in range(self.dataset.numEnt())]
        if head_or_tail == "tail":
            ret_facts = [(head, rel, i, date) for i in range(self.dataset.numEnt())]
        
        if raw_or_fil == "raw":
            ret_facts = [tuple(fact)] + ret_facts
        elif raw_or_fil == "fil":
            ret_facts = [tuple(fact)] + list(set(ret_facts) - self.dataset.all_facts_as_tuples)        
        
        return shredFacts(np.array(ret_facts),hiss,ent_hiss,fact.reshape(1,-1))
    
    def test(self):
        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            settings = ["fil"]
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:
                    if head_or_tail=="head":
                        shiss=np.array(self.dataset.his[self.valid_or_test][0][i]).reshape(1,-1)
                        ent_shiss=np.array(self.dataset.ent_his[self.valid_or_test][0][i]).reshape(1,-1)
                        batch=self.replaceAndShred(fact, shiss,ent_shiss,raw_or_fil, "head")
                        sim_scores = self.model(batch1=batch,batch2=None,train_or_test=self.valid_or_test,ent_type="subs").cpu().data.numpy()
                    if head_or_tail=="tail":
                        ohiss=np.array(self.dataset.his[self.valid_or_test][1][i]).reshape(1,-1)
                        ent_ohiss=np.array(self.dataset.ent_his[self.valid_or_test][1][i]).reshape(1,-1)
                        batch=self.replaceAndShred(fact, ohiss,ent_ohiss,raw_or_fil, "tail")
                        sim_scores = self.model(batch1=batch,batch2=None,train_or_test=self.valid_or_test,ent_type="objs").cpu().data.numpy()
                    rank = self.getRank(sim_scores)
                    self.measure.update(rank, raw_or_fil)
                    
        
        # self.measure.print_()
        # print("~~~~~~~~~~~~~")
        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()    
        
        return self.measure.mrr["fil"]
        
