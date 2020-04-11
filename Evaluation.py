import torch
from GlobalPara import Paras
import numpy as np
from utils.utilGeneral import *
import math
from nltk import Tree


class MyEvaluation:

    @staticmethod
    def get_loss(model):

        pred_list = []
        gold_list = []
        true_positive = 0
        false_positive = 0
        true_negtive = 0
        false_negtive = 0
        g_total = 0.0
        g_correct = 0.0

        for i in range(0, len(Paras.TEST_WORD_SEQs_STANFORD)):
            with torch.no_grad():
                word_seq_ = Paras.TEST_WORD_SEQs_STANFORD[i]
                word_seq_origin = Paras.TEST_WORD_SEQs_STANFORD_Origin[i]
                char_seq_ = Paras.TEST_CHAR_SEQs_STANFORD[i]
                pos_seq_ = Paras.TEST_POS_SEQs_STANFORD[i]
                if len(word_seq_) <= 1 or len(word_seq_) > Paras.MAX_SENT_LEN:
                    continue
                pred = model(word_seq_, char_seq_, pos_seq_, word_seq_origin, i)
                pred_list.append(pred)
                gold_list.append(Paras.TEST_GROUND_STR[i])

                # tp_s, fp_s, tn_s, fn_s, g_t, g_c = model.calculate_attention(word_seq_, char_seq_, pos_seq_, i)
                # true_positive += tp_s
                # false_positive += fp_s
                # true_negtive += tn_s
                # false_negtive += fn_s
                # g_total += g_t
                # g_correct += g_c

        # precision = 0.0
        # recall = 0.0
        # if true_positive+false_positive > 0:
        #     precision = (true_positive+0.0)/(true_positive+false_positive+0.0)
        # if false_negtive+true_positive > 0:
        #     recall = (true_positive+0.0)/(false_negtive+true_positive+0.0)
        # f1 = 0.
        # if precision+recall > 0:
        #     f1 = 2*precision*recall/(precision+recall)
        # print("attention h accuracy:"+str(precision)+","+str(recall)+","+str(f1))
        # g_accuracy = 0.0
        # if g_total > 0:
        #     g_accuracy = g_correct/g_total
        # print("attention g accuracy:"+str(g_accuracy))

        pos_prf, seg_prf, cst_prf, spn_prf = MyEvaluation.evaluate(gold_list, pred_list)

        return seg_prf[2], pos_prf[2], cst_prf[2], spn_prf[2], pred_list

    @staticmethod
    def evaluate(gold_str_list: list, pred_str_list: list):
        """
        :param gold_str_list:   [str]   Ground Truth 树字符串列表
        :param pred_str_list:   [str]   Prediction 树字符串列表
        :return:评估结果字符串
        """
        assert len(gold_str_list) == len(pred_str_list)

        gold_trees = [Tree.fromstring(s) for s in gold_str_list]
        pred_trees = [Tree.fromstring(s) for s in pred_str_list]
        ret = MyEvaluation.evaluate_trees(gold_trees, pred_trees)
        return ret

    @staticmethod
    def evaluate_trees(gold_trees: list, pred_trees: list):
        """
        与evaluate类似，输入是nltk.Tree的列表
        :param gold_trees:
        :param pred_trees:
        :return:
        """
        assert len(gold_trees) == len(pred_trees)

        # 混淆矩阵
        # nums: num_right, num_gold, num_pred
        # num_right:    # true positive
        # num_gold:     # true_positive + # false negative
        # num_pred:     # true_positive + # false positive

        pos_nums = np.array([0, 0, 0])
        seg_nums = np.array([0, 0, 0])
        cst_nums = np.array([0, 0, 0])
        spn_nums = np.array([0, 0, 0])

        # 获取混淆矩阵
        for gold_tree, pred_tree in zip(gold_trees, pred_trees):
            gold_pos, gold_seg, gold_cst, gold_spn = analyze_tree(gold_tree)
            pred_pos, pred_seg, pred_cst, pred_spn = analyze_tree(pred_tree)

            pos_nums += np.array(calc_confusion_num(gold_pos, pred_pos))
            seg_nums += np.array(calc_confusion_num(gold_seg, pred_seg))
            cst_nums += np.array(calc_confusion_num(gold_cst, pred_cst))
            spn_nums += np.array(calc_confusion_num(gold_spn, pred_spn))

        # 计算PRF
        # prf: precision,recall,f1
        pos_prf = calc_prf(pos_nums)
        seg_prf = calc_prf(seg_nums)
        cst_prf = calc_prf(cst_nums)
        spn_prf = calc_prf(spn_nums)

        return pos_prf, seg_prf, cst_prf, spn_prf

def analyze_tree(tree):
    """
    从树中提取label
    :param tree:
    :return:pos_labels, seg_labels, cst_labels, spn_labels
    """
    pos_labels = []
    cst_labels = []
    _recursive_get_labels(node=tree,
                          i=0,
                          pos_labels=pos_labels,
                          cst_labels=cst_labels)
    seg_labels = [span for pos, span in pos_labels]
    spn_labels = [span for cst, span in cst_labels]
    return pos_labels, seg_labels, cst_labels, spn_labels


def _recursive_get_labels(node: Tree, i, pos_labels: list, cst_labels: list):
    if isinstance(node[0], str):
        # preleaf node
        word = node[0]
        j = i + len(word)
        span = i, j
        pos = node.label()
        pos_label = pos, span
        pos_labels.append(pos_label)
        return j
    else:
        # constituent node
        child_j = i
        for child in node:
            child_j = _recursive_get_labels(node=child,
                                            i=child_j,
                                            pos_labels=pos_labels,
                                            cst_labels=cst_labels)
        j = child_j
        span = i, child_j
        cst = node.label()
        cst_label = cst, span
        cst_labels.append(cst_label)
        return j


def calc_confusion_num(golds, preds):
    """
    计算混淆矩阵数值
    :param golds:
    :param preds:
    :return:
    """
    gold_set = set(golds)
    pred_set = set(preds)
    num_right = len(gold_set.intersection(pred_set))
    num_gold = len(gold_set)
    num_pred = len(pred_set)
    return num_right, num_gold, num_pred


def calc_prf(nums):
    """
    计算 precision,recall,f1
    :param nums:    num_right, num_gold, num_pred
    :return:        precision, recall, f1
    """
    assert np.issubdtype(nums.dtype, np.integer)
    right, gold, pred = nums
    precision = right / pred if pred > 0 else 0
    recall = right / gold if gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

