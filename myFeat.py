import torch
import torch.nn as nn
import torch.nn.functional as F
from GlobalPara import Paras
from utils.utilGeneral import *
from myCandidates import *
from utils.utilTorch import *
import os
from my_lstm_tree_cyk import *
from my_bert.modeling_bert import BertModel

class MyFeat(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert_model = BertModel.from_pretrained(Paras.BERT_FOLDER_PATH, output_hidden_states=True, output_attentions=True)

        # freeze the embedding
        for param in list(self.bert_model.embeddings.parameters()):
            param.requires_grad = False
        print("Froze Embedding Layer")
        layer_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for layer_idx in layer_indexes:
            for param in list(self.bert_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            print("Froze Layer: ", layer_idx)

        self.word_embeddings = nn.Embedding(
            Paras.WORD_SIZE, Paras.WORD_EMBEDDING_DIM,
            _weight=torch.from_numpy(Paras.WORD_EMBEDDING_INIT).float())

        self.word_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.pos_embeddings = nn.Embedding(Paras.POS_SIZE, Paras.POS_EMBEDDING_DIM)

        self.W_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.W = nn.Linear(Paras.WORD_EMBEDDING_DIM, Paras.CHAR_EMBEDDING_DIM)

        self.lstmForwardWORD = nn.LSTM(Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM,
                                       Paras.WORD_LSTM_HIDDEN_SIZE,
                                       num_layers=Paras.WORD_LSTM_HIDDEN_LAYER)
        self.h0FWORD = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.c0FWORD = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))

        self.lstmBackwardWORD = nn.LSTM(Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM,
                                        Paras.WORD_LSTM_HIDDEN_SIZE,
                                        num_layers=Paras.WORD_LSTM_HIDDEN_LAYER)
        self.h0BWORD = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.c0BWORD = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.lstmForwardWORD2 = nn.LSTM(Paras.WORD_LSTM_HIDDEN_SIZE*2,
                                       Paras.WORD_LSTM_HIDDEN_SIZE,
                                       num_layers=Paras.WORD_LSTM_HIDDEN_LAYER)
        self.h0FWORD2 = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.c0FWORD2 = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.lstmBackwardWORD2 = nn.LSTM(Paras.WORD_LSTM_HIDDEN_SIZE*2,
                                        Paras.WORD_LSTM_HIDDEN_SIZE,
                                        num_layers=Paras.WORD_LSTM_HIDDEN_LAYER)
        self.h0BWORD2 = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.c0BWORD2 = nn.Parameter(torch.randn(Paras.WORD_LSTM_HIDDEN_LAYER, 1,
                                                Paras.WORD_LSTM_HIDDEN_SIZE))
        self.word_dropout2 = nn.Dropout(Paras.DROPOUT_RATE)

        self.my_lstm_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.my_lstm_forward = MyLSTM_SingleLayer_Tree_CYK()
        self.my_lstm_backward = MyLSTM_SingleLayer_Tree_CYK()

        self.word_lstm_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.span_2_hidden1 = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*0+Paras.WORD_LSTM_HIDDEN_SIZE*2, Paras.Output_HIDDEN_SIZE)
        self.hidden_2_tag = nn.Linear(Paras.Output_HIDDEN_SIZE, Paras.CST_SIZE-1)

        self.cst_embedding = nn.Embedding(Paras.CST_SIZE_FULL+1, Paras.CST_EMBEDDING_DIM)
        # the last one is for the init state. the null is for the end state.
        self.lstmForwardTag = nn.LSTM(Paras.CST_EMBEDDING_DIM, Paras.TAG_LSTM_HIDDEN_SIZE,
                                      num_layers=Paras.TAG_LSTM_HIDDEN_LAYER)

        self.cst_lstm_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.cst_span_2_hidden1 = nn.Linear(Paras.TAG_LSTM_HIDDEN_SIZE, Paras.Output_HIDDEN_SIZE)
        self.cst_hidden_2_tag1 = nn.Linear(Paras.Output_HIDDEN_SIZE, Paras.CST_SIZE_FULL)

    def _get_lstm_features_word_train(self, word_seq_, char_seq_, pos_seq_, sent_ix):

        word_seq_ix = torch.tensor(word_seq_, dtype=torch.long)
        word_seq_embeds = self.word_embeddings(word_seq_ix)

        tmp_list = []
        tmp_list.append(101)
        tmp_list.extend([i for j in char_seq_ for i in j])
        tmp_list.append(102)
        char_seq_ix = torch.tensor([tmp_list], dtype=torch.long)
        char_seq_embeds = self.bert_model(char_seq_ix)[-2][-2].squeeze(0)
        chars_seq_embeds = []
        count = 1
        for chars in char_seq_:
            chars_ixs = []
            for i in range(count, count+len(chars)):
                chars_ixs.append(i)
            chars_seq_ixs = torch.tensor(chars_ixs, dtype=torch.long)
            chars_embeds = torch.index_select(char_seq_embeds, 0, chars_seq_ixs)
            chars_seq_embeds.append(torch.mean(chars_embeds, 0))
            count += len(chars)
        chars_seq_embeds = torch.stack(chars_seq_embeds)
        if word_seq_embeds.size(0) != chars_seq_embeds.size(0):
            print("my error: word_seq length is not equal to chars_seq length...")

        wordChar_seq_embeds = torch.tanh(self.W(self.W_dropout(word_seq_embeds)))+chars_seq_embeds

        pos_seq_ix = torch.tensor(pos_seq_, dtype=torch.long)
        pos_seq_embeds = self.pos_embeddings(pos_seq_ix)

        embeds_forward = torch.cat([wordChar_seq_embeds, pos_seq_embeds], 1)
        embeds_backward = my_reverse_tensor(embeds_forward)
        embeds_forward = self.word_dropout(embeds_forward)
        embeds_backward = self.word_dropout(embeds_backward)

        lstm_input1 = embeds_forward.view(embeds_forward.size(0), 1, Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM)
        lstm_out1, last_state1 = self.lstmForwardWORD(lstm_input1, (self.h0FWORD, self.c0FWORD))
        feat_output1 = lstm_out1.view(embeds_forward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_input2 = embeds_backward.view(embeds_backward.size(0), 1, Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM)
        lstm_out2, last_state2 = self.lstmBackwardWORD(lstm_input2, (self.h0BWORD, self.c0BWORD))
        feat_output2 = lstm_out2.view(embeds_backward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_out = torch.cat([feat_output1, my_reverse_tensor(feat_output2)], 1)

        embeds_forward = self.word_dropout2(lstm_out)
        embeds_backward = self.word_dropout2(my_reverse_tensor(lstm_out))
        lstm_input1 = embeds_forward.view(embeds_forward.size(0), 1, Paras.WORD_LSTM_HIDDEN_SIZE*2)
        lstm_out1, last_state1 = self.lstmForwardWORD2(lstm_input1, (self.h0FWORD2, self.c0FWORD2))
        feat_output1 = lstm_out1.view(embeds_forward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_input2 = embeds_backward.view(embeds_backward.size(0), 1, Paras.WORD_LSTM_HIDDEN_SIZE*2)
        lstm_out2, last_state2 = self.lstmBackwardWORD2(lstm_input2, (self.h0BWORD2, self.c0BWORD2))
        feat_output2 = lstm_out2.view(embeds_backward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_out = torch.cat([feat_output1, my_reverse_tensor(feat_output2)], 1)

        # v_F, hp_f, hn_f, g_f, g_a = self.my_lstm_forward(lstm_out,
        #                            Paras.CTB_H_TRAIN[sent_ix], Paras.CTB_H_P_TRAIN[sent_ix], Paras.CTB_H_N_TRAIN[sent_ix],
        #                            Paras.CTB_G_TRAIN[sent_ix])

        # return feat_output1, feat_output2, v_F, None, hp_f, hn_f, g_f, g_a
        return feat_output1, feat_output2, None, None, None, None, None, None

    def _get_lstm_features_word_test(self, word_seq_, char_seq_, pos_seq_, sent_ix):

        word_seq_ix = torch.tensor(word_seq_, dtype=torch.long)
        word_seq_embeds = self.word_embeddings(word_seq_ix)

        tmp_list = []
        tmp_list.append(101)
        tmp_list.extend([i for j in char_seq_ for i in j])
        tmp_list.append(102)
        char_seq_ix = torch.tensor([tmp_list], dtype=torch.long)
        char_seq_embeds = self.bert_model(char_seq_ix)[-2][-2].squeeze(0)
        chars_seq_embeds = []
        count = 1
        for chars in char_seq_:
            chars_ixs = []
            for i in range(count, count+len(chars)):
                chars_ixs.append(i)
            chars_seq_ixs = torch.tensor(chars_ixs, dtype=torch.long)
            chars_embeds = torch.index_select(char_seq_embeds, 0, chars_seq_ixs)
            chars_seq_embeds.append(torch.mean(chars_embeds, 0))
            count += len(chars)
        chars_seq_embeds = torch.stack(chars_seq_embeds)
        if word_seq_embeds.size(0) != chars_seq_embeds.size(0):
            print("my error: word_seq length is not equal to chars_seq length...")

        wordChar_seq_embeds = torch.tanh(self.W(self.W_dropout(word_seq_embeds)))+chars_seq_embeds

        pos_seq_ix = torch.tensor(pos_seq_, dtype=torch.long)
        pos_seq_embeds = self.pos_embeddings(pos_seq_ix)

        embeds_forward = torch.cat([wordChar_seq_embeds, pos_seq_embeds], 1)
        embeds_backward = my_reverse_tensor(embeds_forward)
        embeds_forward = self.word_dropout(embeds_forward)
        embeds_backward = self.word_dropout(embeds_backward)

        lstm_input1 = embeds_forward.view(embeds_forward.size(0), 1, Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM)
        lstm_out1, last_state1 = self.lstmForwardWORD(lstm_input1, (self.h0FWORD, self.c0FWORD))
        feat_output1 = lstm_out1.view(embeds_forward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_input2 = embeds_backward.view(embeds_backward.size(0), 1, Paras.CHAR_EMBEDDING_DIM+Paras.POS_EMBEDDING_DIM)
        lstm_out2, last_state2 = self.lstmBackwardWORD(lstm_input2, (self.h0BWORD, self.c0BWORD))
        feat_output2 = lstm_out2.view(embeds_backward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_out = torch.cat([feat_output1, my_reverse_tensor(feat_output2)], 1)

        embeds_forward = self.word_dropout2(lstm_out)
        embeds_backward = self.word_dropout2(my_reverse_tensor(lstm_out))
        lstm_input1 = embeds_forward.view(embeds_forward.size(0), 1, Paras.WORD_LSTM_HIDDEN_SIZE*2)
        lstm_out1, last_state1 = self.lstmForwardWORD2(lstm_input1, (self.h0FWORD2, self.c0FWORD2))
        feat_output1 = lstm_out1.view(embeds_forward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_input2 = embeds_backward.view(embeds_backward.size(0), 1, Paras.WORD_LSTM_HIDDEN_SIZE*2)
        lstm_out2, last_state2 = self.lstmBackwardWORD2(lstm_input2, (self.h0BWORD2, self.c0BWORD2))
        feat_output2 = lstm_out2.view(embeds_backward.size(0), Paras.WORD_LSTM_HIDDEN_SIZE)
        lstm_out = torch.cat([feat_output1, my_reverse_tensor(feat_output2)], 1)

        # v_F, hp_f, hn_f, g_f, g_a = self.my_lstm_forward(lstm_out,
        #                            Paras.CTB_H_TEST[sent_ix], Paras.CTB_H_P_TEST[sent_ix], Paras.CTB_H_N_TEST[sent_ix],
        #                            Paras.CTB_G_TEST[sent_ix])

        # return feat_output1, feat_output2, v_F, None, hp_f, hn_f, g_f, g_a
        return feat_output1, feat_output2, None, None, None, None, None, None

    def get_span_features_1(self, word_seq_, char_seq_, pos_seq_, candidates_: MyCandidates_Train, sent_ix):

        tmp_result = self._get_lstm_features_word_train(word_seq_, char_seq_, pos_seq_, sent_ix)
        feat_lstmF = tmp_result[0]
        feat_lstmB = tmp_result[1]
        feat_lstmF = torch.cat([self.h0FWORD2.squeeze(1), feat_lstmF], 0)
        feat_lstmB = torch.cat([self.h0BWORD2.squeeze(1), feat_lstmB], 0)

        spans = candidates_.candidates_list

        # featV_lstmF = tmp_result[2]
        # sent_height = featV_lstmF.size(0)
        # sent_length = featV_lstmF.size(1)
        # feat_size = featV_lstmF.size(2)
        # featV_lstmF = featV_lstmF.contiguous().reshape(sent_height*sent_length, feat_size)
        # vF_ix = torch.tensor([((jx-ix-1)*sent_length+jx-1) for ix, jx in spans], dtype=torch.long)
        # vF = torch.index_select(featV_lstmF, 0, vF_ix)

        hiF_ix = torch.tensor([ix for ix, jx in spans], dtype=torch.long)
        hiF = torch.index_select(feat_lstmF, 0, hiF_ix)
        hjF_ix = torch.tensor([jx for ix, jx in spans], dtype=torch.long)
        hjF = torch.index_select(feat_lstmF, 0, hjF_ix)
        hiB_ix = torch.tensor([len(word_seq_)-jx for ix, jx in spans], dtype=torch.long)
        hiB = torch.index_select(feat_lstmB, 0, hiB_ix)
        hjB_ix = torch.tensor([len(word_seq_)-ix for ix, jx in spans], dtype=torch.long)
        hjB = torch.index_select(feat_lstmB, 0, hjB_ix)

        feat_span = torch.cat([hiF-hjF, hiB-hjB], 1)
        # feat_span = torch.cat([vF], 1)

        feat_mid = torch.relu(self.span_2_hidden1(self.word_lstm_dropout(feat_span)))
        feat_out = self.hidden_2_tag(feat_mid)
        output1 = feat_out

        # feat_mid = torch.tanh(self.span_2_hidden1(self.word_lstm_dropout(feat_span)))
        # feat_out = self.hidden_2_tag(feat_mid)
        # denominator = torch.logsumexp(feat_out, 1)
        # denominator_ex = denominator.unsqueeze(1).expand(-1, Paras.CST_SIZE)
        # output1 = feat_out - denominator_ex

        return output1

    def get_span_features_test(self, word_seq_, char_seq_, pos_seq_, candidates_: MyCandidates_Test, sent_ix):

        tmp_result = self._get_lstm_features_word_test(word_seq_, char_seq_, pos_seq_, sent_ix)
        feat_lstmF = tmp_result[0]
        feat_lstmB = tmp_result[1]
        feat_lstmF = torch.cat([self.h0FWORD2.squeeze(1), feat_lstmF], 0)
        feat_lstmB = torch.cat([self.h0BWORD2.squeeze(1), feat_lstmB], 0)

        spans = candidates_.candidates_list

        # featV_lstmF = tmp_result[2]
        # sent_height = featV_lstmF.size(0)
        # sent_length = featV_lstmF.size(1)
        # feat_size = featV_lstmF.size(2)
        # featV_lstmF = featV_lstmF.contiguous().reshape(sent_height*sent_length, feat_size)
        # vF_ix = torch.tensor([((jx-ix-1)*sent_length+jx-1) for ix, jx in spans], dtype=torch.long)
        # vF = torch.index_select(featV_lstmF, 0, vF_ix)

        hiF_ix = torch.tensor([ix for ix, jx in spans], dtype=torch.long)
        hiF = torch.index_select(feat_lstmF, 0, hiF_ix)
        hjF_ix = torch.tensor([jx for ix, jx in spans], dtype=torch.long)
        hjF = torch.index_select(feat_lstmF, 0, hjF_ix)
        hiB_ix = torch.tensor([len(word_seq_)-jx for ix, jx in spans], dtype=torch.long)
        hiB = torch.index_select(feat_lstmB, 0, hiB_ix)
        hjB_ix = torch.tensor([len(word_seq_)-ix for ix, jx in spans], dtype=torch.long)
        hjB = torch.index_select(feat_lstmB, 0, hjB_ix)

        feat_span = torch.cat([hiF-hjF, hiB-hjB], 1)
        # feat_span = torch.cat([vF], 1)

        feat_mid = torch.relu(self.span_2_hidden1(self.word_lstm_dropout(feat_span)))
        feat_out = self.hidden_2_tag(feat_mid)
        output1 = feat_out

        # feat_mid = torch.tanh(self.span_2_hidden1(self.word_lstm_dropout(feat_span)))
        # feat_out = self.hidden_2_tag(feat_mid)
        # denominator = torch.logsumexp(feat_out, 1)
        # denominator_ex = denominator.unsqueeze(1).expand(-1, Paras.CST_SIZE)
        # output1 = feat_out - denominator_ex

        return output1

    def get_tag_features_1(self, word_seq_, char_seq_, pos_seq_, candidates_: MyCandidates_Train, sent_ix):
        tmp_result = self._get_lstm_features_word_train(word_seq_, char_seq_, pos_seq_, sent_ix)
        feat_lstmF = tmp_result[0]
        feat_lstmB = tmp_result[1]
        feat_lstmF = torch.cat([self.h0FWORD2.squeeze(1), feat_lstmF], 0)
        feat_lstmB = torch.cat([self.h0BWORD2.squeeze(1), feat_lstmB], 0)

        spans = candidates_.candidates_tag_span_list

        # featV_lstmF = tmp_result[2]
        # sent_height = featV_lstmF.size(0)
        # sent_length = featV_lstmF.size(1)
        # feat_size = featV_lstmF.size(2)
        # featV_lstmF = featV_lstmF.contiguous().reshape(sent_height*sent_length, feat_size)
        # vF_ix = torch.tensor([((jx-ix-1)*sent_length+jx-1) for ix, jx in spans], dtype=torch.long)
        # vF = torch.index_select(featV_lstmF, 0, vF_ix)

        hiF_ix = torch.tensor([ix for ix, jx in spans], dtype=torch.long)
        hiF = torch.index_select(feat_lstmF, 0, hiF_ix)
        hjF_ix = torch.tensor([jx for ix, jx in spans], dtype=torch.long)
        hjF = torch.index_select(feat_lstmF, 0, hjF_ix)
        hiB_ix = torch.tensor([len(word_seq_)-jx for ix, jx in spans], dtype=torch.long)
        hiB = torch.index_select(feat_lstmB, 0, hiB_ix)
        hjB_ix = torch.tensor([len(word_seq_)-ix for ix, jx in spans], dtype=torch.long)
        hjB = torch.index_select(feat_lstmB, 0, hjB_ix)

        feat_span = torch.cat([hjF - hiF, hjB - hiB], 1)
        # feat_span = torch.cat([vF], 1)

        tag_input_ix = torch.tensor(candidates_.candidates_tag_seq_input_matrix, dtype=torch.long)

        lstm_input = self.cst_embedding(tag_input_ix)
        h0TAG = torch.unsqueeze(feat_span, 0)
        c0TAG = torch.unsqueeze(feat_span, 0)

        lstm_out, last_state = self.lstmForwardTag(lstm_input, (h0TAG, c0TAG))
        feat_mid = torch.tanh(self.cst_span_2_hidden1(self.cst_lstm_dropout(lstm_out)))
        feat_out = self.cst_hidden_2_tag1(feat_mid)

        denominator = torch.logsumexp(feat_out, 2)
        denominator_ex = denominator.unsqueeze(2).expand(-1, -1, Paras.CST_SIZE_FULL)
        output1 = feat_out - denominator_ex

        return output1

    def get_tag_features_test(self, word_seq_, char_seq_, pos_seq_, span_list, sent_ix):

        tmp_result = self._get_lstm_features_word_test(word_seq_, char_seq_, pos_seq_, sent_ix)
        feat_lstmF = tmp_result[0]
        feat_lstmB = tmp_result[1]
        feat_lstmF = torch.cat([self.h0FWORD2.squeeze(1), feat_lstmF], 0)
        feat_lstmB = torch.cat([self.h0BWORD2.squeeze(1), feat_lstmB], 0)

        spans = span_list

        # featV_lstmF = tmp_result[2]
        # sent_height = featV_lstmF.size(0)
        # sent_length = featV_lstmF.size(1)
        # feat_size = featV_lstmF.size(2)
        # featV_lstmF = featV_lstmF.contiguous().reshape(sent_height*sent_length, feat_size)
        # vF_ix = torch.tensor([((jx-ix-1)*sent_length+jx-1) for ix, jx in spans], dtype=torch.long)
        # vF = torch.index_select(featV_lstmF, 0, vF_ix)

        hiF_ix = torch.tensor([ix for ix, jx in spans], dtype=torch.long)
        hiF = torch.index_select(feat_lstmF, 0, hiF_ix)
        hjF_ix = torch.tensor([jx for ix, jx in spans], dtype=torch.long)
        hjF = torch.index_select(feat_lstmF, 0, hjF_ix)
        hiB_ix = torch.tensor([len(word_seq_) - jx for ix, jx in spans], dtype=torch.long)
        hiB = torch.index_select(feat_lstmB, 0, hiB_ix)
        hjB_ix = torch.tensor([len(word_seq_) - ix for ix, jx in spans], dtype=torch.long)
        hjB = torch.index_select(feat_lstmB, 0, hjB_ix)

        feat_span = torch.cat([hjF - hiF, hjB - hiB], 1)
        # feat_span = torch.cat([vF], 1)

        input_ix = torch.ones([1, len(spans)], dtype=torch.long)
        input_ix = Paras.CST_INIT_IX*input_ix
        lstm_input = self.cst_embedding(input_ix)
        h0TAG = torch.unsqueeze(feat_span, 0)
        c0TAG = torch.unsqueeze(feat_span, 0)

        next_input = lstm_input
        next_h0 = h0TAG
        next_c0 = c0TAG

        tag_list = []
        for index in range(0, 3):
            lstm_out, last_state = self.lstmForwardTag(next_input, (next_h0, next_c0))
            feat_mid = torch.tanh(self.cst_span_2_hidden1(self.cst_lstm_dropout(lstm_out)))
            feat_out = self.cst_hidden_2_tag1(feat_mid)

            denominator = torch.logsumexp(feat_out, 2)
            denominator_ex = denominator.unsqueeze(2).expand(-1, -1, Paras.CST_SIZE_FULL)
            output1 = feat_out - denominator_ex # 1, span_size, cst_size

            tag_select = torch.argmax(output1, 2)
            tag_list.append(tag_select)
            next_input = self.cst_embedding(tag_select)
            next_h0, next_c0 = last_state
        if len(tag_list) == 0:
            print("error: tag_list is null...")

        return torch.cat(tag_list, 0)

