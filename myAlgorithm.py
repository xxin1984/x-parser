import torch
import torch.nn as nn
from myFeat import MyFeat
from myCandidates import *
import numpy as np
from GlobalPara import *
from utils.utilProject import *
import math


class MyAlgorithm(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_model = MyFeat()

    def neg_log_likelihood(self, word_seq_, char_seq_, pos_seq_, labels_, labels_full_, sample_ix):

        candidates_ = MyCandidates_Train(word_seq_, labels_, labels_full_)
        feats = self.feat_model.get_span_features_1(word_seq_, char_seq_, pos_seq_, candidates_, sample_ix)

        # calculating global strucutre
        sent_len = len(word_seq_)
        gold_score = None
        span_score = feats
        for span in labels_.keys():
            i, j = span
            if i+1 < j:
                tmp_ix = candidates_.candidates_dict.get(span)
                label_ix = labels_.get(span)[-1]
                span_score[tmp_ix][labels_.get(span)] = span_score[tmp_ix][label_ix]-1.0
                if gold_score is None:
                    gold_score = span_score[tmp_ix][label_ix]
                else:
                    gold_score = gold_score+span_score[tmp_ix][label_ix]
        span_score, span_score_ix = torch.max(span_score, 1)
        span_score = torch.cat([torch.zeros(sent_len, dtype=torch.float), span_score], 0)

        cyt_ix = Paras.CYK_SPAN_IX.get(sent_len)
        cyt_ix_L = Paras.CYK_SPAN_IX_L.get(sent_len)
        cyt_ix_R = Paras.CYK_SPAN_IX_R.get(sent_len)
        cyt_score = torch.zeros(sent_len, dtype=torch.float)
        for height in range(1, sent_len):
            c_score_ix = torch.tensor(cyt_ix.get(height), dtype=torch.long)
            c_score = torch.index_select(span_score, 0, c_score_ix)
            c_score_L_ix = torch.tensor(cyt_ix_L.get(height), dtype=torch.long)
            c_score_L = torch.index_select(cyt_score, 0, c_score_L_ix)
            c_score_R_ix = torch.tensor(cyt_ix_R.get(height), dtype=torch.long)
            c_score_R = torch.index_select(cyt_score, 0, c_score_R_ix)
            c_score_child = c_score_L+c_score_R
            c_score_child = torch.stack(torch.split(c_score_child, height, 0))
            pred_max_value2, pred_max_ix2 = torch.max(c_score_child, 1)
            c_score = c_score+pred_max_value2
            cyt_score = torch.cat([cyt_score, c_score], 0)
        pred_score = cyt_score[-1]
        loss_global = torch.relu(pred_score-gold_score)/(sent_len-1.)

        ############################################
        # test global
        ############################################
        # span_score = torch.log(1.000001 - torch.exp(torch.transpose(feats, 0, 1)[Paras.CST_2_IX.get("NULL")]))
        # span_score = torch.cat([torch.zeros(sent_len, dtype=torch.float), span_score], 0)
        # sent_len = len(word_seq_)
        # cyt_ix = Paras.CYK_SPAN_IX.get(sent_len)
        # cyt_ix_L = Paras.CYK_SPAN_IX_L.get(sent_len)
        # cyt_ix_R = Paras.CYK_SPAN_IX_R.get(sent_len)
        # cyt_score = torch.zeros(sent_len, dtype=torch.float)
        # label_score = -torch.ones(sent_len, dtype=torch.long)
        # for height in range(1, sent_len):
        #     c_score_ix = torch.tensor(cyt_ix.get(height), dtype=torch.long)
        #     c_score = torch.index_select(span_score, 0, c_score_ix)
        #     c_score_L_ix = torch.tensor(cyt_ix_L.get(height), dtype=torch.long)
        #     c_score_L = torch.index_select(cyt_score, 0, c_score_L_ix)
        #     c_score_R_ix = torch.tensor(cyt_ix_R.get(height), dtype=torch.long)
        #     c_score_R = torch.index_select(cyt_score, 0, c_score_R_ix)
        #
        #     c_score_child = c_score_L+c_score_R
        #     c_score_child = torch.stack(torch.split(c_score_child, height, 0))
        #     pred_max_value2, pred_max_ix2 = torch.max(c_score_child, 1)
        #     c_score = c_score+pred_max_value2
        #     cyt_score = torch.cat([cyt_score, c_score], 0)
        #     label_score = torch.cat([label_score, pred_max_ix2], 0)
        #
        # span_list = []
        # span_dict = {}
        # index = 0
        # for height in range(0, sent_len):
        #     for i in range(0, sent_len - height):
        #         span_list.append((i, i + height + 1))
        #         span_dict[(i, i + height + 1)] = index
        #         index = index + 1
        #
        # pred_spans = []
        # span_stack = []
        # span_stack.append((0, sent_len))
        # while len(span_stack) > 0:
        #     span = span_stack.pop(-1)
        #     i, j = span
        #     pred_divide_ix = label_score[span_dict.get(span)].item()
        #     sub_span_L = (i, i+1+pred_divide_ix)
        #     sub_span_R = (i+1+pred_divide_ix, j)
        #     if i+1+pred_divide_ix > i+1:
        #         pred_spans.append(sub_span_L)
        #         span_stack.append(sub_span_L)
        #     if j > i+1+pred_divide_ix+1:
        #         pred_spans.append(sub_span_R)
        #         span_stack.append(sub_span_R)
        # correct_span = 0.
        # incorrect_span = 0.
        # for span in pred_spans:
        #     if labels_.get(span) is not None:
        #         correct_span += 1.
        #     else:
        #         incorrect_span += 1.
        #
        # print("span accuracy: "+str(correct_span/(correct_span+incorrect_span)))

        ############################################

        # calculating tag
        nll_tag = torch.tensor(0.)
        feats_tag = self.feat_model.get_tag_features_1(word_seq_, char_seq_, pos_seq_, candidates_, sample_ix)
        mask_ix_t = torch.tensor(candidates_.candidates_tag_mask_tensor, dtype=torch.uint8)
        log_likelihood_tag_list = torch.masked_select(feats_tag, mask_ix_t)
        if log_likelihood_tag_list.size(0) > 0:
            nll_tag = -torch.mean(log_likelihood_tag_list, 0)

        return nll_tag + loss_global
        # return loss_global + nll_tag

    def forward(self, word_seq_, char_seq_, pos_seq_, word_seq_origin_, sample_ix):

        # step1 generate structure
        # generate the nltk tree by CYK
        candidates_ = MyCandidates_Test(word_seq_)
        ix_pairs_dict = candidates_.candidates_dict
        feats = self.feat_model.get_span_features_test(word_seq_, char_seq_, pos_seq_, candidates_, sample_ix)
        preds = feats.cpu().numpy()  # span, score of being a cst
        ix_pair_2_divide = {}
        ix_pair_2_score_accumul = {}
        for k in range(0, len(word_seq_)):
            ix_pair_2_score_accumul[(k, k + 1)] = 0.
        for k in range(2, len(word_seq_) + 1):
            for i in range(0, len(word_seq_) - k + 1):
                ix_pair = i, i + k
                span_ix = ix_pairs_dict.get((i, i+k))
                new_score = -1000000000.0
                j_record = -1
                pred_score = np.max(preds[span_ix])
                for j in range(i + 1, i + k):
                    step_p = pred_score+ix_pair_2_score_accumul.get((i, j))+ix_pair_2_score_accumul.get((j, i + k))
                    if step_p > new_score:
                        new_score = step_p
                        j_record = j
                ix_pair_2_divide[ix_pair] = j_record
                ix_pair_2_score_accumul[ix_pair] = new_score
        if len(ix_pair_2_divide) == 0:
            print("error: ix_pair_2_divide is 0...")

        pred_root = self._gen_tree(ix_pair_2_divide, (0, len(word_seq_)), word_seq_origin_)

        # step 2 generating fake tags
        node_list = my_post_order(pred_root)
        span_list = my_post_order_span_ix(pred_root)
        span_tag_list = []

        for span in span_list:
            if ix_pairs_dict.get(span) is None:
                span_tag_list.append("tag")
                continue
            pred = preds[ix_pairs_dict.get(span)]

            high_score = -10000000000000.
            high_id = -1
            tmp_score_star = 0.0
            tmp_score_unstar = 0.0
            for tag_ix in range(0, len(pred)):
                if Paras.CST_IX_2_IS_STAR.get(tag_ix) is True:
                    tmp_score_star += pred[tag_ix]
                    if pred[tag_ix] > high_score:
                        high_score = pred[tag_ix]
                        high_id = tag_ix
                else:
                    tmp_score_unstar += pred[tag_ix]
                    if pred[tag_ix] > high_score:
                        high_score = pred[tag_ix]
                        high_id = tag_ix

            #if tmp_score_star > tmp_score_unstar:

            if Paras.CST_IX_2_IS_STAR.get(high_id) is True:
                span_tag_list.append("tag*")
            else:
                span_tag_list.append("tag")

        for node, tag in zip(node_list, span_tag_list):
            node.set_label(tag)
        pred_root = my_recover_tree(pred_root)

        # step 3 generating full tags
        node_list = my_post_order(pred_root)
        span_list = my_post_order_span_ix(pred_root)

        if len(node_list) == 0 or len(span_list) == 0:
            print("error: node list or span list = 0...")

        tag_feat = self.feat_model.get_tag_features_test(word_seq_, char_seq_, pos_seq_, span_list, sample_ix)

        tag_feat_np = tag_feat.cpu().numpy()
        tag_feat_np = np.transpose(tag_feat_np)

        span_tag_list = []
        for i in range(0, len(tag_feat_np)):
            start, end = span_list[i]
            tag_list = tag_feat_np[i]
            step_list = []
            if start+1 == end:
                step_list.append(Paras.IX_2_POS.get(pos_seq_[start]).split("_")[0])
            for tag_id in tag_list:
                if tag_id == Paras.CST_2_IX_FULL.get("NULL"):
                    break
                else:
                    step_list.append(Paras.IX_2_CST_FULL.get(tag_id))
            span_tag_list.append(step_list)

        if len(span_tag_list) == 0:
            print("error: span_tag_list is null...")

        node_addr_2_tag_list = {}
        for node, tag_list in zip(node_list, span_tag_list):
            node_addr_2_tag_list[id(node)] = tag_list

        for node, span in zip(node_list, span_list):
            if my_is_leaf(node) is False and my_is_pre_leaf(node) is False:
                for k in range(0, len(node)):
                    child = node[k]
                    tag_list = node_addr_2_tag_list.get(id(child))

                    if len(tag_list) == 0:
                        child.set_label("NULL")
                    else:
                        if len(tag_list) == 1:
                            child.set_label(tag_list[0])
                        else:
                            child.set_label(tag_list[0])
                            tmp_node = child
                            for ii in range(1, len(tag_list)):
                                tmp_node = Tree(tag_list[ii], [tmp_node])
                            node.pop(k)
                            node.insert(k, tmp_node)

        root_tag_list = span_tag_list[-1]
        if len(root_tag_list) == 0:
            pred_root.set_label("NULL")
        else:
            pred_root.set_label(root_tag_list[0])
            for ii in range(1, len(root_tag_list)):
                pred_root = Tree(root_tag_list[ii], [pred_root])

        return my_oneline(pred_root)

    def _gen_tree(self, _ix_pair_2_divide, _ix_pair, _word_seq_origin):
        i, j = _ix_pair
        if _ix_pair_2_divide.get(_ix_pair) is None:
            # i+1 == j
            node = Tree("tag", [_word_seq_origin[i]])
            return node
        else:
            m = _ix_pair_2_divide.get(_ix_pair)
            node1 = self._gen_tree(_ix_pair_2_divide, (i, m), _word_seq_origin)
            node2 = self._gen_tree(_ix_pair_2_divide, (m, j), _word_seq_origin)
            node = Tree("tag", [node1, node2])
            return node
