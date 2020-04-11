import torch.nn as nn
from torch.nn import init
from utils.utilTorch import*
from GlobalPara import *
from utils.utilProject import *


class CombTwoNodes_CYK(nn.Module):

    def __init__(self):

        self.head_num = 2

        super(CombTwoNodes_CYK, self).__init__()

        self.cst_linear = nn.Linear(4*Paras.MY_HLSTM_INPUT_DIM, Paras.Output_HIDDEN_SIZE)
        self.cst_output = nn.Linear(Paras.Output_HIDDEN_SIZE, 1)
        self.cst_dropout = nn.Dropout(Paras.DROPOUT_RATE)

        self.weight_hh_llrr_u = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*2, self.head_num * Paras.MY_HLSTM_INPUT_DIM)
        self.weight_hh_llrr_i = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*2, self.head_num * Paras.MY_HLSTM_INPUT_DIM)
        self.weight_hh_llrr_fl = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*2, self.head_num * Paras.MY_HLSTM_INPUT_DIM)
        self.weight_hh_llrr_fr = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*2, self.head_num * Paras.MY_HLSTM_INPUT_DIM)
        self.weight_hh_llrr_o = nn.Linear(Paras.MY_HLSTM_INPUT_DIM*2, self.head_num * Paras.MY_HLSTM_INPUT_DIM)

    def forward(self, input_h, input_c, level, length, CTB_H, CTB_H_P, CTB_H_N, CTB_G):

        span_ix_L = torch.tensor(Paras.SUB_SPAN_IX_L.get(length).get(level), dtype=torch.long)
        span_ix_R = torch.tensor(Paras.SUB_SPAN_IX_R.get(length).get(level), dtype=torch.long)
        span_shallow_ix_L = torch.tensor(Paras.SUB_SPAN_SHALLOW_IX_L.get(length).get(level), dtype=torch.long)
        span_shallow_ix_R = torch.tensor(Paras.SUB_SPAN_SHALLOW_IX_R.get(length).get(level), dtype=torch.long)

        left_h = torch.index_select(input_h, 0, span_ix_L)
        right_h = torch.index_select(input_h, 0, span_ix_R)
        left_c = torch.index_select(input_c, 0, span_ix_L)
        right_c = torch.index_select(input_c, 0, span_ix_R)
        left_sl_h = torch.index_select(input_h, 0, span_shallow_ix_L)
        right_sl_h = torch.index_select(input_h, 0, span_shallow_ix_R)

        input_length = left_h.size(0)
        node_length = length-level
        candi_length = level

        # step 1 calculate the score and select path

        feat2 = torch.cat([left_h, right_h, left_sl_h, right_sl_h], 1)
        feat1 = torch.cat([left_h, right_h], 1)

        score2 = self.cst_output(torch.tanh(self.cst_linear(self.cst_dropout(feat2)))).squeeze(1)
        score2 = torch.sigmoid(score2)
        score2 = torch.stack(torch.split(score2, candi_length, 0))
        score_gate = score2
        pred_max_value2, pred_max_ix2 = torch.max(score2, 1)

        feat1 = torch.stack(torch.split(feat1, candi_length, 0))
        mask = pred_max_ix2.unsqueeze(1).expand(node_length, 2*Paras.MY_HLSTM_INPUT_DIM).unsqueeze(1)
        feat1 = torch.gather(feat1, 1, mask).squeeze(1)

        u = torch.tanh(self.weight_hh_llrr_u(feat1))
        i = torch.sigmoid(self.weight_hh_llrr_i(feat1))
        fl = torch.sigmoid(self.weight_hh_llrr_fl(feat1))
        fr = torch.sigmoid(self.weight_hh_llrr_fr(feat1))
        o = torch.sigmoid(self.weight_hh_llrr_o(feat1))

        left_c = torch.stack(torch.split(left_c, candi_length, 0))
        right_c = torch.stack(torch.split(right_c, candi_length, 0))
        mask_c = pred_max_ix2.unsqueeze(1).expand(node_length, Paras.MY_HLSTM_INPUT_DIM).unsqueeze(1)
        left_c = torch.gather(left_c, 1, mask_c).squeeze(1)
        right_c = torch.gather(right_c, 1, mask_c).squeeze(1)
        left_c_mul = left_c.repeat(1, self.head_num)
        right_c_mul = right_c.repeat(1, self.head_num)

        c_1 = fl * left_c_mul + fr * right_c_mul + i * u
        h_1 = o * torch.tanh(c_1)

        score2 = torch.cat([pred_max_value2.unsqueeze(1), (1.-pred_max_value2).unsqueeze(1)], 1)
        score2_plus = score2.view(node_length, self.head_num, 1).expand(node_length, self.head_num,
                                                                        Paras.MY_HLSTM_INPUT_DIM).reshape(node_length,
                                                                                                           self.head_num * Paras.MY_HLSTM_INPUT_DIM)
        c_1 = c_1 * score2_plus
        h_1 = h_1 * score2_plus

        c_1 = torch.sum(torch.stack(torch.split(c_1, Paras.MY_HLSTM_INPUT_DIM, 1)), 0)
        h_1 = torch.sum(torch.stack(torch.split(h_1, Paras.MY_HLSTM_INPUT_DIM, 1)), 0)

        # step 2 supervise attention

        head_gold_p = torch.tensor(CTB_H_P, dtype=torch.uint8)
        head_score_gold_p = torch.masked_select(pred_max_value2, head_gold_p)
        head_gold_n = torch.tensor(CTB_H_N, dtype=torch.uint8)
        head_score_gold_n = torch.masked_select(pred_max_value2, head_gold_n)

        ix_p = []
        for i in range(0, head_gold_p.size(0)):
            if head_gold_p[i] == 1:
                ix_p.append(i)
        ix_p_tensor = torch.tensor(ix_p, dtype=torch.long)
        ix_p_len = len(ix_p)

        gate_loss = torch.tensor([])
        gate_a = torch.tensor([])
        if ix_p_len > 0:
            score_gate = torch.index_select(score_gate, 0, ix_p_tensor)
            gold_gate_ix = torch.index_select(torch.tensor(CTB_G, dtype=torch.long), 0, ix_p_tensor)

            score_gold_gate = torch.gather(score_gate, 1, gold_gate_ix.unsqueeze(1))
            score_gold_gate = score_gold_gate.expand(ix_p_len, candi_length)

            gate_a, gate_a_ix = torch.max(torch.relu(score_gate-score_gold_gate), 1)
            gate_loss = torch.relu(score_gate-score_gold_gate).reshape(1, -1).squeeze(0)

        return h_1, c_1, head_score_gold_p, head_score_gold_n, gate_loss, gate_a


class MyLSTM_SingleLayer_Tree_CYK(nn.Module):

    def __init__(self):
        super(MyLSTM_SingleLayer_Tree_CYK, self).__init__()
        self.VLSTM = CombTwoNodes_CYK()
        self.c_dropout = nn.Dropout(Paras.DROPOUT_RATE)
        self.h_dropout = nn.Dropout(Paras.DROPOUT_RATE)

    def forward(self, input_seq_i, CTB_H, CTB_H_P, CTB_H_N, CTB_G):
        # step 0 init

        input_seq = input_seq_i
        input_length = input_seq.size(0)

        #step 1 开始生成每个节点的树结构特征

        VLSTM_h_list = []
        VLSTM_c_list = []

        next_h = input_seq_i
        next_c = input_seq_i

        VLSTM_h_list.append(self.h_dropout(next_h))
        VLSTM_c_list.append(self.c_dropout(next_c))

        current_h_1D = self.h_dropout(next_h)
        current_c_1D = self.c_dropout(next_c)

        head_p = []
        head_n = []
        gate = []
        gate_a = []

        for i in range(1, min(input_length, Paras.MY_HEIGHT_LIMIT)):
            next_h, next_c, head_p_s, head_n_s, gate_s, gate_a_s = \
                self.VLSTM(current_h_1D, current_c_1D, i, input_length,
                           CTB_H.get(i), CTB_H_P.get(i), CTB_H_N.get(i),
                           CTB_G.get(i))

            missed_tensor = torch.zeros([i, Paras.MY_HLSTM_INPUT_DIM], dtype=torch.float)
            next_h = torch.cat([missed_tensor, next_h], 0)
            next_c = torch.cat([missed_tensor, next_c], 0)

            VLSTM_h_list.append(self.h_dropout(next_h))
            VLSTM_c_list.append(self.c_dropout(next_c))

            current_h_1D = torch.cat([current_h_1D, self.h_dropout(next_h)], 0)
            current_c_1D = torch.cat([current_c_1D, self.c_dropout(next_c)], 0)

            if head_p_s.size(0) > 0:
                head_p.append(head_p_s)
            if head_n_s.size(0) > 0:
                head_n.append(head_n_s)
            if gate_s.size(0) > 0:
                gate.append(gate_s)
            if gate_a_s.size(0) > 0:
                gate_a.append(gate_a_s)

        VLSTM_out = torch.stack(VLSTM_h_list)

        Head_p_out = None
        Head_n_out = None
        Gate_out = None
        Gate_a_out = None

        if len(head_p) > 0:
            Head_p_out = torch.cat(head_p, 0)
        if len(head_n) > 0:
            Head_n_out = torch.cat(head_n, 0)
        if len(gate) > 0:
            Gate_out = torch.cat(gate, 0)
        if len(gate_a) > 0:
            Gate_a_out = torch.cat(gate_a, 0)

        return VLSTM_out, Head_p_out, Head_n_out, Gate_out, Gate_a_out
        # return VLSTM_out, None, None, None, None
