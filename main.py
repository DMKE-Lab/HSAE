# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os

from dataset import Dataset
from params import Params
from tester import Tester
from trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset',help='Dataset',type=str,default='icews14',choices=['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model',help='Model',type=str,default='HSAE_simple',choices=['HSAE_complex', 'HSAE_simple', 'HSAE_distmult'])
parser.add_argument('-ne',help='Number of epochs',type=int,default=500,choices=[500])  #500
parser.add_argument('-bsize',help='Batch size',type=int,default=512,choices=[512])  #512
parser.add_argument('-lr',help='Learning rate',type=float,default=0.001,choices=[0.001])
parser.add_argument('-reg_lambda',help='L2 regularization parameter',type=float,default=0.0,choices=[0.0])
parser.add_argument('-emb_dim',help='Embedding dimension',type=int,default=100,choices=[100])
parser.add_argument('-neg_ratio',help='Negative ratio',type=int,default=500,choices=[5, 500])  #500
parser.add_argument('-dropout',help='Dropout probability',type=float,default=0.4,choices=[0.0, 0.2, 0.4])
parser.add_argument('-save_each',help='Save model and validate each K epochs',type=int,default=20,choices=[20,100])  #20
parser.add_argument('-se_prop',help='Static embedding proportion',type=float,default=0.68)
parser.add_argument('-beta', help='alphi', type=float, default=0.5)
parser.add_argument('-e_epoch', help='K epochs', type=int, default=20)


args = parser.parse_args()

dataset = Dataset(args.dataset, args.bsize)

params = Params(ne=args.ne,bsize=args.bsize,
                lr=args.lr,
                reg_lambda=args.reg_lambda,
                emb_dim=args.emb_dim,
                neg_ratio=args.neg_ratio,
                dropout=args.dropout,
                save_each=args.save_each,
                se_prop=args.se_prop)

trainer = Trainer(dataset, params, args.model)
# trainer.train()

# validating the trained models
validation_idx = [
    str(int(args.save_each * (i + 1)))
    for i in range(args.ne // args.save_each)
]
best_mrr = -1.0
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_(
) + "_"

for idx in validation_idx:
    model_path = model_prefix + idx + ".chkpnt"
    tester = Tester(dataset, model_path, "valid", args.model)
    mrr = tester.test()
    print(f'idx:{idx},mrr:{mrr}')
    if mrr > best_mrr:
        best_mrr = mrr
        best_index = idx
        print(f'best_mrr:{best_mrr},indx:{best_index}')

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"
tester = Tester(dataset, model_path, "test", args.model)
tester.test()

# # validating the trained models. we seect the model that has the best validation performance as the fina model
# validation_idx = [
#     str(int(args.save_each * (i + 1)))                                                              
#     for i in range(args.ne // args.save_each)
# ]
# best_mrr = -1.0
# best_index = '0'  #'0'
# model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_(
# ) + "_"

# # epoch_mrr=[]
# # test_epoch_mrr=[]
# # epoch=[]
# # for i in range(20,520,20):
# #     epoch.append(i)

# # for idx in validation_idx:
# #     model_path = model_prefix + idx + ".chkpnt"
#     # tester = Tester(dataset, model_path, "valid", args.model)
#     # mrr = tester.test()
#     # epoch_mrr.append(mrr)
#     # if mrr > best_mrr:
#     #     best_mrr = mrr
#     #     best_index = idx
        
#     # test_tester = Tester(dataset, model_path, "test", args.model)
#     # test_mrr = test_tester.test()
#     # test_epoch_mrr.append(test_mrr)

# # testing the best chosen model on the test set
# best_index = '500'
# print("Best epoch: " + best_index)
# model_path = model_prefix + best_index + ".chkpnt"
# tester = Tester(dataset, model_path, "test", args.model)
# tester.test()

# # print(trainer.tol_loss)
# # print(epoch_mrr)
# # print(test_epoch_mrr)

# # fig, ax1 = plt.subplots()
# # ax1.plot(epoch, epoch_mrr, color="blue", label="valid_MRR")
# # ax1.plot(epoch, test_epoch_mrr, color="green", label="test_MRR")
# # ax1.set_xlabel("epoch")
# # ax1.set_ylabel("MRR")

# # ax2 = ax1.twinx()
# # ax2.plot(epoch, trainer.tol_loss, color="red", label="Loss")
# # ax2.set_ylabel("Loss")

# # fig.legend()
# # plt.show()
