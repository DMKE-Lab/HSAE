# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from params import Params
from HSAE_complex import HSAE_complex
from HSAE_distmult import HSAE_distmult
from HSAE_simple import HSAE_simple


class Trainer:
    def __init__(self, dataset, params, model_name):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params).cuda())
        self.dataset = dataset
        self.params = params
        self.tol_loss = []
        
    def train(self, early_stop=False):
        self.model.train()
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.lr, 
            weight_decay=self.params.reg_lambda
        ) #weight_decay corresponds to L2 regularization
        
        loss_f = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()
            
            while not last_batch:
                optimizer.zero_grad()
                batch1,batch2 = self.dataset.nextBatch(self.params.bsize,neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()
                scores=self.model(batch1,batch2)
                
                ###Added for softmax####
                heads=torch.cat((batch1[0],batch2[0]),0)
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio+1)
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                
            # 转换为localtime
            localtime = time.localtime(time.time())
            # 利用strftime()函数重新格式化时间
            print(str(time.strftime('%Y:%m:%d %H:%M:%S', localtime))+'   '+str(time.time() - start)) # 返回当前时间：2021:09:09 19:17:29
            
            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")
            
            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)
                self.tol_loss.append(total_loss)
            
    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # torch.save(self.model.state_dict(), directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")
        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")
        
