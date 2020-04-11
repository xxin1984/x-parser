from GlobalPara import Paras
import numpy as np


class MyCandidates_Train:

    def __init__(self, word_seq_, labels_, labels_full_):
        super().__init__()

        # for generating structures
        self.candidates_list = None
        self.candidates_dict = None
        self.init_candidates(word_seq_, labels_)

        # for generating cst tags
        self.candidates_tag_span_list = None
        self.candidates_tag_span_dict = None
        self.candidates_tag_mask_tensor = None
        self.candidates_tag_seq_input_matrix = None
        self.init_candidates_tag(word_seq_, labels_full_)

    def init_candidates_tag(self, word_seq_, labels_full_):
        self.candidates_tag_span_list = []
        self.candidates_tag_span_dict = {}
        index = 0
        for i in range(0, len(word_seq_)):
            for j in range(i+1, len(word_seq_)+1):
                if (j == i+1) or (labels_full_.get((i, j)) is not None):
                    self.candidates_tag_span_list.append((i, j))
                    self.candidates_tag_span_dict[(i, j)] = index
                    index = index+1

        self.candidates_tag_mask_tensor = np.zeros([Paras.DECODER_LENGTH, len(self.candidates_tag_span_list), Paras.CST_SIZE_FULL], dtype=np.int32)
        self.candidates_tag_seq_input_matrix = np.zeros([Paras.DECODER_LENGTH, len(self.candidates_tag_span_list)], dtype=np.int32)
        for i, j in self.candidates_tag_span_list:
            span_ix = self.candidates_tag_span_dict.get((i, j))
            if labels_full_.get((i, j)) is not None:
                labels = labels_full_.get((i, j))
                tag = Paras.CST_INIT_IX
                for k in range(0, len(labels)):
                    self.candidates_tag_mask_tensor[k][span_ix][labels[k]] = 1
                    self.candidates_tag_seq_input_matrix[k][span_ix] = tag
                    tag = labels[k]
                self.candidates_tag_mask_tensor[len(labels)][span_ix][Paras.CST_2_IX_FULL.get("NULL")] = 1
                self.candidates_tag_seq_input_matrix[len(labels)][span_ix] = tag
            else:
                self.candidates_tag_mask_tensor[0][span_ix][Paras.CST_2_IX_FULL.get("NULL")] = 1
                self.candidates_tag_seq_input_matrix[0][span_ix] = Paras.CST_INIT_IX

    def init_candidates(self, word_seq_, labels_):
        self.candidates_list = []
        self.candidates_dict = {}
        index = 0
        for height in range(1, len(word_seq_)):
            for i in range(0, len(word_seq_)-height):
                self.candidates_list.append((i, i+height+1))
                self.candidates_dict[(i, i+height+1)] = index
                index = index+1


class MyCandidates_Test:

    def __init__(self, word_seq_):
        super().__init__()

        # for generating structures
        self.candidates_list = None
        self.candidates_dict = None
        self.init_candidates(word_seq_)

    def init_candidates(self, word_seq_):
        self.candidates_list = []
        self.candidates_dict = {}
        index = 0
        for height in range(1, len(word_seq_)):
            for i in range(0, len(word_seq_)-height):
                self.candidates_list.append((i, i+height+1))
                self.candidates_dict[(i, i+height+1)] = index
                index = index+1
