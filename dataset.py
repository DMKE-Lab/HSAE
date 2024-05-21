# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from scripts import shredFacts


class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, 
                 ds_name,batch_size):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.batch_size=batch_size
        self.ent2id = {}
        self.rel2id = {}
        self.time2id={}
        self.year2id={}
        self.month2id={}
        self.day2id={}

        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}
        
        self.ent_his = {"train":self.get_ent_his(self.data["train"], self.data["valid"],self.data["test"])[0],
                    "valid":self.get_ent_his(self.data["train"], self.data["valid"],self.data["test"])[1],
                    "test":self.get_ent_his(self.data["train"], self.data["valid"],self.data["test"])[2]}
      
      
        self.his = {"train":self.get_his(self.data["train"], self.data["valid"],self.data["test"])[0],
                    "valid":self.get_his(self.data["train"], self.data["valid"],self.data["test"])[1],
                    "test":self.get_his(self.data["train"], self.data["valid"],self.data["test"])[2]}
        
  
        
        self.start_batch = 0
        self.all_facts_as_tuples = None
        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])
            self.his[spl]=np.array(self.his[spl],dtype=object)
            self.ent_his[spl]=np.array(self.ent_his[spl],dtype=object)
        self.ent_shiss,self.ent_ohiss=self.get_ent_pad(self.ent_his["train"][0]), self.get_ent_pad(self.ent_his["train"][1])
           
        self.shiss,self.ohiss=self.get_pad(self.his["train"][0]), self.get_pad(self.his["train"][1])
        
    def get_sort_data(self, t=0):
        self.train_data, self.train_his = self.data["train"], [self.his["train"][0], self.his["train"][1]]
        idxs = self.get_sorted_idx()
        self.train_data, self.train_his = self.train_data[idxs], [self.train_his[0][idxs], self.train_his[1][idxs]]


    def get_ent_pad(self, his_data):
        start_batch = 0
        his_pad = []
        while start_batch + self.batch_size < len(his_data):
            hiss = self.ent_padding(his_data[start_batch:start_batch +self.batch_size])
            start_batch += self.batch_size
        hiss = self.ent_padding(his_data[self.start_batch:])
        his_pad.extend(hiss)
        return his_pad

    def ent_padding(self, hiss):
        batch_size, maxlen = len(hiss), max(map(len, hiss))
        hiss_new = []
        for i, his in enumerate(hiss):
            curlen = len(his)
            padding = [self.numEnt() for _ in range(maxlen - curlen)]
            hiss_new.append(his + padding)
        return np.array(hiss_new)

    def get_pad(self, his_data):
        start_batch = 0
        his_pad = []
        while start_batch + self.batch_size < len(his_data):
            hiss = self.padding(his_data[start_batch:start_batch +self.batch_size])
            start_batch += self.batch_size
        hiss = self.padding(his_data[self.start_batch:])
        his_pad.extend(hiss)
        return his_pad

    def padding(self, hiss):
        batch_size, maxlen = len(hiss), max(map(len, hiss))
        hiss_new = []
        for i, his in enumerate(hiss):
            curlen = len(his)
            padding = [self.numRel() for _ in range(maxlen - curlen)]
            hiss_new.append(his + padding)
        return np.array(hiss_new)


    def readFile(self, filename):

        with open(filename, "r",encoding='UTF-8') as f:
            data = f.readlines()
        
        facts = []
        for line in data:
            # strip方法用于移除字符串头尾指定的字符（默认为空格）或字符序列
            elements = line.strip().split("\t")
            # elements[3]=elements[3].replace('#','0')
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            time_id= self.getTimeID(elements[3])
            year_id=self.getYearID(elements[3])
            month_id=self.getMonthID(elements[3])
            day_id=self.getDayID(elements[3])
            timestamp = elements[3]
            facts.append([head_id, rel_id, tail_id,time_id])
            
        return facts
    
    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)
    def numTime(self):
        
        return len(self.time2id)
    def numYear(self):
        return len(self.year2id)
    def numMonth(self):
        return len(self.month2id)
    def numDay(self):
        return len(self.day2id)
    
    def getEntID(self,ent_name):
        if ent_name in self.ent2id:
            return self.ent2id[ent_name] 
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    # 年-月级
    def getTimeID(self,time_date):
        if time_date[:7] in self.time2id:
            return self.time2id[time_date[:7]]
        self.time2id[time_date[:7]]=len(self.time2id)
        return self.time2id[time_date[:7]]

    # 年级
    # def getTimeID(self,time_date):
    #     if time_date[:4] in self.time2id:
    #         return self.time2id[time_date[:4]]
    #     self.time2id[time_date[:4]]=len(self.time2id)
    #     return self.time2id[time_date[:4]]
#____________________________________________________________    
    
    def getYearID(self,time_date):
        if time_date[:4] in self.year2id:
            return self.year2id[time_date[:4]]
        self.year2id[time_date[:4]]=len(self.year2id)
        return self.year2id[time_date[:4]]

    def getMonthID(self,time_date):
        if time_date[5:7] in self.month2id:
            return self.month2id[time_date[5:7]]
        self.month2id[time_date[5:7]]=len(self.month2id)
        return self.month2id[time_date[5:7]]
    
    def getDayID(self,time_date):
        if time_date[8:11] in self.day2id:
            return self.day2id[time_date[8:11]]
        self.day2id[time_date[8:11]]=len(self.day2id)
        return self.day2id[time_date[8:11]]
#____________________________________________________________________

    def get_ent_his(self, train_quas, valid_quas, test_quas):
        quas = train_quas + valid_quas
        
        subs_dict = {(ent,rel):set() for ent in self.ent2id.values() for rel in self.rel2id.values()}
        objs_dict = {(ent,rel):set() for ent in self.ent2id.values() for rel in self.rel2id.values()}      

        for qua in quas:
            subs_dict[(qua[0],qua[1])].update([qua[2]])
            objs_dict[(qua[2],qua[1])].update([qua[0]])
              
        train_subs_his, train_objs_his = [], []
        for qua in train_quas:
            train_subs_his.append(list(subs_dict[(qua[0],qua[1])]))
            train_objs_his.append(list(objs_dict[(qua[2],qua[1])]))
              
        valid_subs_his, valid_objs_his = [], []
        for qua in valid_quas:
            valid_subs_his.append(list(subs_dict[(qua[0],qua[1])]))
            valid_objs_his.append(list(objs_dict[(qua[2],qua[1])]))
              
        test_subs_his, test_objs_his = [], []
        for qua in test_quas:
            if (qua[0],qua[1]) in subs_dict.keys():
                test_subs_his.append(list(subs_dict[(qua[0],qua[1])]))
            else:
                test_subs_his.append([])
            if (qua[2],qua[1]) in objs_dict.keys():
                test_objs_his.append(list(objs_dict[(qua[2],qua[1])]))
            else:
                test_objs_his.append([])

        return [train_subs_his,train_objs_his], [valid_subs_his, valid_objs_his], [test_subs_his, test_objs_his]    


    def get_his(self, train_quas, valid_quas, test_quas):
        quas = train_quas + valid_quas
        subs_dict = {ent: set() for ent in self.ent2id.values()}
        objs_dict = {ent: set() for ent in self.ent2id.values()}

        for qua in quas:
            subs_dict[qua[0]].add(qua[1])
            objs_dict[qua[2]].add(qua[1])

        train_subs_his, train_objs_his = [], []
        for qua in train_quas:
            train_subs_his.append(list(subs_dict[qua[0]]))
            train_objs_his.append(list(objs_dict[qua[2]]))

        valid_subs_his, valid_objs_his = [], []
        for qua in valid_quas:
            valid_subs_his.append(list(subs_dict[qua[0]]))
            valid_objs_his.append(list(objs_dict[qua[2]]))

        test_subs_his, test_objs_his = [], []

        for qua in test_quas:
            if qua[0] in subs_dict.keys():
                test_subs_his.append(list(subs_dict[qua[0]]))
            else:
                test_subs_his.append([])
            if qua[2] in objs_dict.keys():
                test_objs_his.append(list(objs_dict[qua[2]]))
            else:
                test_objs_his.append([])

        return [train_subs_his,train_objs_his], [valid_subs_his, valid_objs_his], [test_subs_his, test_objs_his]    

    def get_sorted_idx(self):
        assert len(self.train_his[0]) == len(self.train_his[1])
        sub_his, obj_his = self.train_his[0], self.train_his[1]
        sub_len, obj_len = -np.asarray(list(map(len, sub_his))), -np.asarray(
            list(map(len, obj_his)))
        length = np.add(sub_len, obj_len)
        idxs = np.lexsort((obj_len, sub_len, length), )

        return idxs
  
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch:]
            shiss, ohiss = self.shiss[self.start_batch:], self.ohiss[self.start_batch:]
            ent_shiss, ent_ohiss = self.ent_shiss[self.start_batch:], self.ent_ohiss[self.start_batch:]
            if len(ret_facts) % 2 != 0:
                ret_facts = np.append(ret_facts,ret_facts[-1].reshape(1, -1),axis=0)
                shiss = np.append(shiss, shiss[-1].reshape(1, -1), axis=0)
                ohiss = np.append(ohiss, ohiss[-1].reshape(1, -1), axis=0)
                ent_shiss = np.append(ent_shiss, ent_shiss[-1].reshape(1, -1), axis=0)
                ent_ohiss = np.append(ent_ohiss, ent_ohiss[-1].reshape(1, -1), axis=0)
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch:self.start_batch +batch_size]
            shiss, ohiss = self.shiss[self.start_batch:self.start_batch +batch_size], self.ohiss[self.start_batch:self.start_batch +batch_size]
            ent_shiss, ent_ohiss = self.ent_shiss[self.start_batch:self.start_batch +batch_size], self.ent_ohiss[self.start_batch:self.start_batch +batch_size]
            self.start_batch += batch_size
        return ret_facts, shiss, ohiss,ent_shiss,ent_ohiss
    

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio
            
            facts[s1:e1,0] = (facts[s1:e1,0] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            facts[s2:e2,2] = (facts[s2:e2,2] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            
        return facts
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()
        return facts1, facts2
    
    def nextBatch(self, batch_size, neg_ratio=1):
        """
        Returns:
            [heads, rels, tails, dates,ohiss],[heads, rels, tails, dates,shiss]
        """
        bp_facts,shiss,ohiss,ent_shiss,ent_ohiss = self.nextPosBatch(batch_size)
        batch1,batch2 = self.addNegFacts2(bp_facts, neg_ratio)
        return  shredFacts(batch1,ohiss,ent_ohiss,bp_facts),shredFacts(batch2,shiss,ent_shiss,bp_facts)
    
    def wasLastBatch(self):
        return (self.start_batch == 0)
            
