from utils.utilGeneral import *
from utils.utilTree import *
import numpy as np
import math
import json
from my_bert.tokenization_bert import *

class Paras:

    WORD_LSTM_HIDDEN_SIZE = None
    WORD_LSTM_HIDDEN_LAYER = None
    TAG_LSTM_HIDDEN_SIZE = None
    TAG_LSTM_HIDDEN_LAYER = None
    Output_HIDDEN_SIZE = None
    DECODER_LENGTH = None
    DROPOUT_RATE = None
    MAX_SENT_LEN = None

    MY_HLSTM_INPUT_DIM = None
    MY_HEIGHT_LIMIT = None

    CHAR_EMBEDDING_INIT: np.ndarray = None
    WORD_EMBEDDING_INIT: np.ndarray = None
    CHAR_EMBEDDING_DIM = None
    WORD_EMBEDDING_DIM = None
    POS_EMBEDDING_DIM = None
    CST_EMBEDDING_DIM = None

    CHAR_2_IX: dict = None
    WORD_2_IX: dict = None
    POS_2_IX: dict = None
    CST_2_IX: dict = None
    CST_2_IX_FULL: dict = None
    HEAD_CAT_2_IX: dict = None

    IX_2_CHAR: dict = None
    IX_2_WORD: dict = None
    IX_2_POS: dict = None
    IX_2_CST: dict = None
    IX_2_CST_FULL: dict = None
    IX_2_HEAD_CAT: dict = None

    CST_IX_2_IS_STAR: dict = None

    WORD_SIZE = None
    CHAR_SIZE = None
    POS_SIZE = None
    CST_SIZE = None
    CST_SIZE_FULL = None
    HEAD_CAT_SIZE = None

    CST_INIT_IX = None

    TRAIN_WORD_SEQs = None
    TRAIN_CHAR_SEQs = None
    TRAIN_POS_SEQs = None
    TRAIN_LABELs = None
    TRAIN_LABELs_FULL = None

    TEST_WORD_SEQs = None
    TEST_CHAR_SEQs = None
    TEST_POS_SEQs = None
    TEST_LABELs = None
    TEST_LABELs_FULL = None

    TEST_WORD_SEQs_STANFORD = None
    TEST_WORD_SEQs_STANFORD_Origin = None
    TEST_CHAR_SEQs_STANFORD = None
    TEST_POS_SEQs_STANFORD = None
    TEST_GROUND_STR = None

    BERT_FOLDER_PATH = None

    SUB_SPAN_IX_L: dict = None
    SUB_SPAN_IX_R: dict = None
    SUB_SPAN_SHALLOW_IX_L: dict = None
    SUB_SPAN_SHALLOW_IX_R: dict = None

    CYK_SPAN_IX: dict = None
    CYK_SPAN_IX_L: dict = None
    CYK_SPAN_IX_R: dict = None

    # ctb
    CTB_RATE = 0.5
    CTB_SEQ_LIST: list = None

    CTB_H_TRAIN: list = None
    CTB_H_P_TRAIN: list = None
    CTB_H_N_TRAIN: list = None
    CTB_G_TRAIN: list = None
    CTB_G_RVS_TRAIN: list = None

    CTB_H_TEST: list = None
    CTB_H_P_TEST: list = None
    CTB_H_N_TEST: list = None
    CTB_G_TEST: list = None
    CTB_G_RVS_TEST: list = None


    @staticmethod
    def init():

        Paras.WORD_LSTM_HIDDEN_SIZE = 100
        Paras.WORD_LSTM_HIDDEN_LAYER = 1
        Paras.TAG_LSTM_HIDDEN_SIZE = 200
        Paras.TAG_LSTM_HIDDEN_LAYER = 1
        Paras.Output_HIDDEN_SIZE = 128
        Paras.DECODER_LENGTH = 3
        Paras.DROPOUT_RATE = 0.15
        Paras.MAX_SENT_LEN = 100

        Paras.MY_HLSTM_INPUT_DIM = 200
        Paras.MY_HEIGHT_LIMIT = 60  # max length of a sentence

        Paras.CHAR_EMBEDDING_DIM = 768
        Paras.POS_EMBEDDING_DIM = 32
        Paras.WORD_EMBEDDING_DIM = 80
        Paras.CST_EMBEDDING_DIM = 32

        # map文件相关的char和word已经全部转全角

        Paras.CHAR_2_IX = {}
        Paras.IX_2_CHAR = {}
        ix = 0
        lines = my_open_file("data/pretrained_model/vocab.txt")
        for line in lines:
            Paras.CHAR_2_IX[line] = ix
            Paras.IX_2_CHAR[ix] = line
            ix += 1

        Paras.WORD_2_IX = my_read_ix("data/word_map1.txt")
        Paras.POS_2_IX = my_read_ix("data/pos_map.txt")
        Paras.CST_2_IX = my_read_ix("data/cst_map.txt")
        Paras.CST_2_IX_FULL = my_read_ix("data/cst_map_full.txt")
        Paras.HEAD_CAT_2_IX = my_read_ix("data/head_categories.txt")

        Paras.IX_2_WORD = my_read_ix_reverse("data/word_map1.txt")
        Paras.IX_2_POS = my_read_ix_reverse("data/pos_map.txt")
        Paras.IX_2_CST = my_read_ix_reverse("data/cst_map.txt")
        Paras.IX_2_CST_FULL = my_read_ix_reverse("data/cst_map_full.txt")
        Paras.IX_2_HEAD_CAT = my_read_ix_reverse("data/head_categories.txt")

        Paras.CST_IX_2_IS_STAR = {}
        for i in range(0, len(Paras.IX_2_CST)):
            cst_tag = Paras.IX_2_CST.get(i)
            if cst_tag[-1] == "*":
                Paras.CST_IX_2_IS_STAR[i] = True

        Paras.CHAR_SIZE = len(Paras.CHAR_2_IX)
        Paras.WORD_SIZE = len(Paras.WORD_2_IX)
        Paras.POS_SIZE = len(Paras.POS_2_IX)
        Paras.CST_SIZE = len(Paras.CST_2_IX)
        Paras.CST_SIZE_FULL = len(Paras.CST_2_IX_FULL)
        Paras.HEAD_CAT_SIZE = len(Paras.HEAD_CAT_2_IX)

        Paras.CST_INIT_IX = Paras.CST_SIZE_FULL
        Paras.WORD_EMBEDDING_INIT = word_embedding_init("data/word_embedding.txt")

        Paras.BERT_FOLDER_PATH = "data/pretrained_model"

        init_sub_span_ix()
        init_cyk_span_ix()

        Paras.CTB_H_TRAIN = []
        Paras.CTB_H_P_TRAIN = []
        Paras.CTB_H_N_TRAIN = []
        Paras.CTB_G_TRAIN = []
        Paras.CTB_G_RVS_TRAIN = []

        Paras.CTB_H_TEST = []
        Paras.CTB_H_P_TEST = []
        Paras.CTB_H_N_TEST = []
        Paras.CTB_G_TEST = []
        Paras.CTB_G_RVS_TEST = []

        # init_train("data/debug.txt", "data/debug_gold.txt")
        # init_test("data/debug.txt", "data/debug_stanford.txt", "data/debug_gold.txt")
        # init_ctb("data/debug.txt", Paras.CTB_H_TRAIN, Paras.CTB_H_P_TRAIN, Paras.CTB_H_N_TRAIN,
        #          Paras.CTB_G_TRAIN, Paras.CTB_G_RVS_TRAIN)
        # init_ctb("data/debug.txt", Paras.CTB_H_TEST, Paras.CTB_H_P_TEST, Paras.CTB_H_N_TEST,
        #          Paras.CTB_G_TEST, Paras.CTB_G_RVS_TEST)

        init_train("data_r/train10_r.txt", "data_r/train9_r.txt")
        init_test("data_r/test10_r.txt", "data_r/test9_stanford_r.txt", "data_r/test9_r.txt")
        # init_ctb("data/train10.txt", Paras.CTB_H_TRAIN, Paras.CTB_H_P_TRAIN, Paras.CTB_H_N_TRAIN,
        #          Paras.CTB_G_TRAIN, Paras.CTB_G_RVS_TRAIN)
        # init_ctb("data/test10.txt", Paras.CTB_H_TEST, Paras.CTB_H_P_TEST, Paras.CTB_H_N_TEST,
        #          Paras.CTB_G_TEST, Paras.CTB_G_RVS_TEST)

        print("init over...")


def init_cyk_span_ix():
    Paras.CYK_SPAN_IX = {}
    Paras.CYK_SPAN_IX_L = {}
    Paras.CYK_SPAN_IX_R = {}
    for L in range(2, Paras.MAX_SENT_LEN+1):
        h2ixs = {}
        h2ix_left = {}
        h2ix_right = {}
        for h in range(1, L):
            tmp_list = []
            tmp_list_L = []
            tmp_list_R = []
            index0 = int(L * h - (h - 1) * h / 2)
            for j in range(0, L - h):
                tmp_list.append(index0 + j)
                for k in range(j + 1, j + h + 1):
                    # left_tag = (j, k)
                    # right_tag = (k, j + h + 1)
                    left_ix = int(L * (k - j - 1) - (k - j - 2) * (k - j - 1) / 2) + j
                    right_ix = int(L * (j + h - k) - (j + h - k - 1) * (j + h - k) / 2) + k
                    tmp_list_L.append(left_ix)
                    tmp_list_R.append(right_ix)
            h2ixs[h] = tmp_list
            h2ix_left[h] = tmp_list_L
            h2ix_right[h] = tmp_list_R
        Paras.CYK_SPAN_IX[L] = h2ixs
        Paras.CYK_SPAN_IX_L[L] = h2ix_left
        Paras.CYK_SPAN_IX_R[L] = h2ix_right


def init_sub_span_ix():
    Paras.SUB_SPAN_IX_L = {}
    Paras.SUB_SPAN_IX_R = {}
    Paras.SUB_SPAN_SHALLOW_IX_L = {}
    Paras.SUB_SPAN_SHALLOW_IX_R = {}
    for L in range(2, Paras.MAX_SENT_LEN+1):
        sentence_left = {}
        sentence_right = {}
        sentence_shallow_left = {}
        sentence_shallow_right = {}
        for i in range(1, L):
            layer_left = []
            layer_right = []
            layer_shallow_left = []
            layer_shallow_right = []
            for j in range(i, L):
                for k in range(j - i + 1, j + 1):
                    layer_left.append((k - j + i - 1) * L + k - 1)
                    layer_right.append((j - k) * L + j)
                    layer_shallow_left.append((j-i))
                    layer_shallow_right.append(j)
            sentence_left[i] = layer_left
            sentence_right[i] = layer_right
            sentence_shallow_left[i] = layer_shallow_left
            sentence_shallow_right[i] = layer_shallow_right
        Paras.SUB_SPAN_IX_L[L] = sentence_left
        Paras.SUB_SPAN_IX_R[L] = sentence_right
        Paras.SUB_SPAN_SHALLOW_IX_L[L] = sentence_shallow_left
        Paras.SUB_SPAN_SHALLOW_IX_R[L] = sentence_shallow_right


def init_train(train_file, train_file_full):

    Paras.TRAIN_WORD_SEQs = []
    Paras.TRAIN_CHAR_SEQs = []
    Paras.TRAIN_POS_SEQs = []
    Paras.TRAIN_LABELs = []
    Paras.TRAIN_LABELs_FULL = []

    tokenizer = BertTokenizer.from_pretrained(Paras.BERT_FOLDER_PATH)

    lines = my_open_file(train_file)
    for line in lines:
        root = Tree.fromstring(line)
        word_list = []
        char_list = []
        for leaf in root.leaves():
            word = my_strB2Q(leaf)
            word_id = Paras.WORD_2_IX.get(word, Paras.WORD_2_IX.get("NULL"))
            word_list.append(word_id)

            c_list = tokenizer.encode(leaf)
            char_list.append(c_list)

        Paras.TRAIN_WORD_SEQs.append(word_list)
        Paras.TRAIN_CHAR_SEQs.append(char_list)

        pos_list = []
        for node in my_post_order(root):
            if my_is_pre_leaf(node):
                pos_tag = node.label()
                pos_id = Paras.POS_2_IX.get(pos_tag)
                pos_list.append(pos_id)
        Paras.TRAIN_POS_SEQs.append(pos_list)

        IXPair_2_Tags = {}
        node_list = my_post_order(root)
        span_list = my_post_order_span_ix(root)
        for node, span in zip(node_list, span_list):
            if my_is_pre_leaf(node) is False:
                label = Paras.CST_2_IX.get(node.label())
                if IXPair_2_Tags.get(span) is None:
                    IXPair_2_Tags[span] = []
                IXPair_2_Tags.get(span).append(label)

        Paras.TRAIN_LABELs.append(IXPair_2_Tags)

    lines = my_open_file(train_file_full)
    for line in lines:
        root = Tree.fromstring(line)
        IXPair_2_Tags = {}
        node_list = my_post_order(root)
        span_list = my_post_order_span_ix(root)
        for node, span in zip(node_list, span_list):
            if my_is_pre_leaf(node) is False:
                label = Paras.CST_2_IX_FULL.get(node.label())
                if IXPair_2_Tags.get(span) is None:
                    IXPair_2_Tags[span] = []
                IXPair_2_Tags.get(span).append(label)

        Paras.TRAIN_LABELs_FULL.append(IXPair_2_Tags)


def _get_tree_mask(node_: Tree, left_p_mask_, right_p_mask_, node_2_span_):
    if my_is_pre_leaf(node_) is not True:
        i, j = node_2_span_.get(id(node_))
        left_p_mask_[(j-i-1, j-1)] = True
        right_p_mask_[(j-i-1, j-1)] = True

        i_left, j_left = node_2_span_.get(id(node_[0]))
        for jj in range(j_left+1, j):
            left_p_mask_[(jj-i-1, jj-1)] = True
        i_right, j_right = node_2_span_.get(id(node_[1]))
        for ii in range(i+1, i_right):
            right_p_mask_[(j-ii-1, j-1)] = True
        _get_tree_mask(node_[0], left_p_mask_, right_p_mask_, node_2_span_)
        _get_tree_mask(node_[1], left_p_mask_, right_p_mask_, node_2_span_)


def init_ctb(train_file, CTB_H, CTB_H_P, CTB_H_N, CTB_G, CTB_G_RVS):

    print("read ctb begin...")

    lines = my_open_file(train_file)
    for line in lines:
        line = my_delete_unary(line)
        root = Tree.fromstring(line)
        leaves = root.leaves()
        length = len(leaves)
        height = min(Paras.MY_HEIGHT_LIMIT, length)

        node_list = my_post_order(root)
        span_list = my_post_order_span_ix(root)
        node_2_span = {id(node): span for node, span in zip(node_list, span_list)}

        left_mask_dict = {}
        right_mask_dict = {}
        _get_tree_mask(root, left_mask_dict, right_mask_dict, node_2_span)

        left_mask = np.zeros([length, length], dtype=np.int32)
        right_mask = np.zeros([length, length], dtype=np.int32)
        for key in left_mask_dict.keys():
            i, j = key
            left_mask[i][j] = 1
        for key in right_mask_dict.keys():
            i, j = key
            right_mask[i][j] = 1

        gate_mask = {}
        gate_rev_mask = {}

        for i in range(1, height):
            gate_step = np.zeros([length-i], dtype=np.int32)
            gate_rev_step = np.zeros([length-i], dtype=np.int32)

            for j in range(i, length):
                gate_step[j-i] = i-1
                gate_rev_step[j-i] = 0
                for k in range(1, i):
                    if left_mask[k][j] == 1 and right_mask[k][j] == 1:
                        gate_step[j-i] = i-1-k
                    if left_mask[k][j-i+k] == 1 and right_mask[k][j-i+k] == 1:
                        gate_rev_step[j-i] = k
            gate_mask[i] = gate_step
            gate_rev_mask[i] = gate_rev_step

        label_mask = np.zeros([length, length], dtype=np.int32)
        for node, span in zip(node_list, span_list):
            label = node.label()
            label_ix = 0
            # if label[-1] == "r":
            #     label_ix = 1
            # if label[-1] == "l":
            #     label_ix = 2
            # if label[-1] == "*" and label[-2] == "l":
            #     label_ix = 3
            # if label[-1] == "*" and label[-2] == "r":
            #     label_ix = 4
            if label[-1] == "r" or label[-1] == "l" or \
                    (label[-1] == "*" and label[-2] == "l") or \
                    (label[-1] == "*" and label[-2] == "r"):
                label_ix = 1

            i, j = span
            label_mask[j-i-1][j-1] = label_ix

        head_mask = {}
        head_p_mask = {}
        head_n_mask = {}

        for i in range(1, height):
            head_step = np.zeros([length-i], dtype=np.int)
            head_p_step = np.zeros([length-i], dtype=np.int)
            head_n_step = np.ones([length-i], dtype=np.int)
            for j in range(i, length):
                head_step[j-i] = label_mask[i][j]
                if label_mask[i][j] > 0:
                    head_p_step[j-i] = 1
                    head_n_step[j-i] = 0

            head_mask[i] = head_step
            head_p_mask[i] = head_p_step
            head_n_mask[i] = head_n_step

        CTB_H.append(head_mask)
        CTB_H_P.append(head_p_mask)
        CTB_H_N.append(head_n_mask)
        CTB_G.append(gate_mask)
        CTB_G_RVS.append(gate_rev_mask)

    print("read ctb end...")


def init_test(test_file, test_stanford_file, gold_file):

    Paras.TEST_WORD_SEQs = []
    Paras.TEST_CHAR_SEQs = []
    Paras.TEST_POS_SEQs = []
    Paras.TEST_LABELs = []
    Paras.TEST_LABELs_FULL = []

    tokenizer = BertTokenizer.from_pretrained(Paras.BERT_FOLDER_PATH)

    lines = my_open_file(test_file)
    for line in lines:
        root = Tree.fromstring(line)
        word_list = []
        char_list = []
        for leaf in root.leaves():
            word = my_strB2Q(leaf)
            word_id = Paras.WORD_2_IX.get(word, Paras.WORD_2_IX.get("NULL"))
            word_list.append(word_id)
            c_list = tokenizer.encode(leaf)
            char_list.append(c_list)

        Paras.TEST_WORD_SEQs.append(word_list)
        Paras.TEST_CHAR_SEQs.append(char_list)

        pos_list = []
        for node in my_post_order(root):
            if my_is_pre_leaf(node):
                pos_tag = node.label()
                pos_id = Paras.POS_2_IX.get(pos_tag)
                pos_list.append(pos_id)
        Paras.TEST_POS_SEQs.append(pos_list)

        IXPair_2_Tags = {}
        node_list = my_post_order(root)
        span_list = my_post_order_span_ix(root)

        for node, span in zip(node_list, span_list):
            if my_is_pre_leaf(node) is False:
                label = Paras.CST_2_IX.get(node.label())
                if Paras.CST_2_IX.get(node.label()) is None:
                    print("error: node label is null...")
                if IXPair_2_Tags.get(span) is None:
                    IXPair_2_Tags[span] = []
                IXPair_2_Tags.get(span).append(label)

        Paras.TEST_LABELs.append(IXPair_2_Tags)

    lines = my_open_file(gold_file)
    for line in lines:
        root = Tree.fromstring(line)
        IXPair_2_Tags = {}
        node_list = my_post_order(root)
        span_list = my_post_order_span_ix(root)

        for node, span in zip(node_list, span_list):
            if my_is_pre_leaf(node) is False:
                label = Paras.CST_2_IX_FULL.get(node.label())
                if Paras.CST_2_IX_FULL.get(node.label()) is None:
                    print("error: node label is null...")
                if IXPair_2_Tags.get(span) is None:
                    IXPair_2_Tags[span] = []
                IXPair_2_Tags.get(span).append(label)

        Paras.TEST_LABELs_FULL.append(IXPair_2_Tags)

    Paras.TEST_WORD_SEQs_STANFORD = []
    Paras.TEST_WORD_SEQs_STANFORD_Origin = []
    Paras.TEST_CHAR_SEQs_STANFORD = []
    Paras.TEST_POS_SEQs_STANFORD = []
    Paras.TEST_GROUND_STR = []

    lines = my_open_file(test_stanford_file)
    for line in lines:
        elements = line.split(" ")
        word_list = []
        word_list_origin = []
        char_list = []
        pos_list  = []
        for element in elements:
            word0, pos = element.split("_")
            pos = pos+"_t"
            word_list_origin.append(word0)
            word = my_strB2Q(word0)
            word_id = Paras.WORD_2_IX.get(word, Paras.WORD_2_IX.get("NULL"))
            word_list.append(word_id)
            pos_id = Paras.POS_2_IX.get(pos)
            pos_list.append(pos_id)
            c_list = tokenizer.encode(word0)
            char_list.append(c_list)

        Paras.TEST_WORD_SEQs_STANFORD.append(word_list)
        Paras.TEST_WORD_SEQs_STANFORD_Origin.append(word_list_origin)
        Paras.TEST_CHAR_SEQs_STANFORD.append(char_list)
        Paras.TEST_POS_SEQs_STANFORD.append(pos_list)

    lines = my_open_file(gold_file)
    for line in lines:
        Paras.TEST_GROUND_STR.append(line)


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def word_embedding_init(word_emb_file):

    scale = np.sqrt(3.0 / Paras.WORD_EMBEDDING_DIM)
    embedding = np.empty([Paras.WORD_SIZE, Paras.WORD_EMBEDDING_DIM])
    embedding[0, :] = np.random.uniform(-scale, scale, Paras.WORD_EMBEDDING_DIM)

    lines = my_open_file(word_emb_file)
    for line in lines:
        items = line.split(" ")
        vec = []
        for i in range(1, len(items)):
            vec.append(float(items[i]))
        embedding[Paras.WORD_2_IX.get(items[0]), :] = norm2one(np.array(vec))

    return embedding


if __name__ == '__main__':
    Paras.init()

