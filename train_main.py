import torch
from GlobalPara import Paras
from Evaluation import MyEvaluation
import torch.optim as optim
from myAlgorithm import MyAlgorithm
import os
from utils.utilTorch import *
from utils.utilGeneral import *
from time import time
import math

CUDA = 0
DIRNAME = "output/t7040"
env = Environment(DIRNAME, CUDA)

Paras.init()

model = MyAlgorithm().cuda()

# check_file = "output/t3210/checkpoints/e73.ckpt"
# my_load_checkpoint(check_file, model)

model.eval()
seg_f1, pos_f1, cst_f1, spn_prf1, preds = MyEvaluation.get_loss(model)
content = " seg f1:" + str(seg_f1) + \
          " pos f1:" + str(pos_f1) + \
          " cst f1:" + str(cst_f1) + \
          " spn f1:" + str(spn_prf1) + "\n"

print(content)
env.save_print(content)
env.save_prediction(-1, preds)


optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-8)
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

last_loss = 10000.
last_lr = 0.1

for epoch in range(10000):

    if epoch == 31:
        # freeze the embedding
        # for param in list(model.feat_model.bert_model.embeddings.parameters()):
        #     param.requires_grad = True
        # print("fine-tune Embedding Layer")
        layer_indexes = [6, 7, 8, 9, 10, 11]
        for layer_idx in layer_indexes:
            for param in list(model.feat_model.bert_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = True
            print("fine-tune Layer: ", layer_idx)

    content = "epoch:"+str(epoch)
    print(content)
    env.save_print(content)

    loss_list = []

    t0 = time()
    for i in range(0, len(Paras.TRAIN_WORD_SEQs)):
        if i % 1000 == 0:
            print(i)

        word_seq_ = Paras.TRAIN_WORD_SEQs[i]
        char_seq_ = Paras.TRAIN_CHAR_SEQs[i]
        pos_seq_ = Paras.TRAIN_POS_SEQs[i]
        label_ = Paras.TRAIN_LABELs[i]
        label_full_ = Paras.TRAIN_LABELs_FULL[i]
        if len(word_seq_) <= 1 or len(word_seq_) > Paras.MAX_SENT_LEN:
            continue
        #step 1
        model.zero_grad()
        model.train()
        #step 2 run forward pass
        loss = model.neg_log_likelihood(word_seq_, char_seq_, pos_seq_, label_, label_full_, i)
        loss_list.append(loss.item())

        if math.isnan(loss.item()):
            print(i)
        #step 3 compute the gradient
        loss.backward()
        # my_clip(model)
        optimizer.step()
    if epoch == 30:
        env.save_checkpoint(epoch, model)

    c_loss = my_avg_list(loss_list)
    print("loss in the train set:"+str(c_loss))

    # if c_loss > last_loss:
    last_lr = last_lr*(1-0.05)
    if last_lr < 0.0005:
        last_lr = 0.0005
    optimizer = my_set_lr(optimizer, last_lr)
    last_loss = c_loss

    model.eval()
    seg_f1, pos_f1, cst_f1, spn_prf1, preds = MyEvaluation.get_loss(model)
    content = " seg f1:" + str(seg_f1) +\
              " pos f1:" + str(pos_f1) + \
              " cst f1:" + str(cst_f1) + \
              " spn f1:" + str(spn_prf1) + "\n"

    print(content)
    env.save_print(content)
    env.save_prediction(epoch, preds)
    t1 = time()
    print("time:"+str(t1-t0))

