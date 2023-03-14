# -*- coding: utf-8 -*-
"""
@CreateTime :       2022/12/28 21:25
@Author     :       Qingpeng Wen
@File       :       layers.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2022/12/28 23:35
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from utils import operation as op

MASK_VALUE = -2 ** 32 + 1
nth = 2

class NullOp(nn.Module):
    def forward(self, input):
        return input

# TODO: Related to SOG-Fusion
class FusionLayer(nn.Module):
    def __init__(self, x_size, y_size, dropout_rate, fusion_type, size_out):
        """
                    x_size = x
                    y_size = y
                    size_out = 384
        """
        super(FusionLayer, self).__init__()
        self.__x_size = x_size
        self.__y_size = y_size
        self.__fusion_type = fusion_type
        self.__dropout_layer = nn.Dropout(dropout_rate)
        self.size_out = size_out

        wh1 = torch.Tensor(size_out, x_size)
        self.wh1 = nn.Parameter(wh1)
        wh2 = torch.Tensor(size_out, y_size)
        self.wh2 = nn.Parameter(wh2)
        weight_sigmoid = torch.Tensor(size_out, size_out * 2)

        assert fusion_type in ["add", "rate", "linear", "bilinear", "weight_sigmoid"]
        if fusion_type == "bilinear":
            self.__bilinear = nn.Bilinear(x_size, y_size, 1)
        elif fusion_type == "rate":
            self.__fusion_rate = nn.Parameter(torch.randn(1))
        elif fusion_type == "linear":
            self.__fusion_linear = nn.Linear(x_size, 1)
        elif fusion_type == "weight_sigmoid":
            self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.wh1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wh2, a=math.sqrt(5))

    def forward(self, x, y=None, dropout=True):
        """
            x1 = tanh(x)
            y1 = tanh(y)
            f = σ[(W1 * x + W2 * y)]
            x2 = f * x + x
            y2 = (1 - f) * y + y
            fusion =f * x2 + (1 - f) * y2
        """
        assert len(x.shape) == 2 and x.size(1) == self.__x_size

        if y is None:
            return self.__linear_layer(self.__dropout_layer(x) if dropout else x)

        assert len(y.shape) == 2 and y.size(1) == self.__y_size

        x0 = torch.tanh(x)
        y0 = torch.tanh(y)

        x1 = torch.mm(x0, self.wh1.t())
        y1 = torch.mm(y0, self.wh2.t())

        if self.__fusion_type == "bilinear":
            fusion_rates = self.__bilinear(x1, y1)
        elif self.__fusion_type == "rate":
            fusion_rates = self.__fusion_rate
        elif self.__fusion_type == "linear":
            fusion_rates = self.__fusion_linear(
                x1)
        elif self.__fusion_type == "weight_sigmoid":

            # with weight
            h = torch.cat((x1, y1), dim=1)
            fusion_rates = torch.matmul(h, self.weight_sigmoid.t())

        f = torch.sigmoid(fusion_rates)

        x2 = x0
        y2 = y0

        for i in range(2, nth + 1):
            x2 = f * x2 + x
            y2 = (1 - f) * y2 + y

        fusion = f * x2 + (1 - f) * y2
        if dropout:
            fusion = self.__dropout_layer(fusion)

        return fusion  # [batch,hidden]

# TODO: Related to SF-Update
class SF_FusionLayer(nn.Module):
    def __init__(self, hid_size, dropout_rate, size_out):
        """
                    x_size = x
                    y_size = y
                    size_out = 384
        """
        super(SF_FusionLayer, self).__init__()
        self.__dropout_layer = nn.Dropout(dropout_rate)
        self.size_out = size_out

        # Weights hidden state modality 1
        self.W0 = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.W1 = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.W2 = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out, size_out * 2)

        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

    def forward(self, x, y, dropout=True):
        """
            x1 = tanh(x)
            y1 = tanh(y)
            f = σ[W3 * tanh(W1 * x + W2 * y)]
            x2 = f * x1 + x
            y2 = (1 - f) * y1 + y
            fusion = f * x2 + (1 - f) * y2
        """

        x0 = torch.tanh(x)
        y0 = torch.tanh(y)

        x1 = self.W0(x0)
        y1 = self.W1(y0)

        h = torch.tanh(x1 + y1)
        fusion_rates = self.W2(h)

        f = torch.sigmoid(fusion_rates)

        x2 = x0
        y2 = y0

        for i in range(2, nth+1):
            x2 = f * x2 + x
            y2 = (1 - f) * y2 + y

        fusion = f * x2 + (1 - f) * y2
        if dropout:
            fusion = self.__dropout_layer(fusion)

        return fusion

# TODO: Related to ID-Update
class ID_FusionLayer(nn.Module):
    def __init__(self, hid_size, dropout_rate, size_out):
        """
                    x_size = x
                    y_size = y
                    size_out = 384
        """
        super(ID_FusionLayer, self).__init__()
        self.__dropout_layer = nn.Dropout(dropout_rate)
        self.size_out = size_out

        self.W3 = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.W4 = nn.Linear(in_features=hid_size, out_features=hid_size, bias=True)
        self.W5 = nn.Linear(in_features=hid_size, out_features=1, bias=None)

    def forward(self, x, y, dropout=False):
        """
            x1 = tanh(x)
            y1 = tanh(y)
            f = softmax[W3 * (W1 * x + W2 * y)]
            fusion =f * x + (1 - f) * Y
        """
        x0 = torch.tanh(x)
        y0 = torch.tanh(y)
        x1 = self.W3(x)
        y1 = self.W4(y)

        fusion_rates = self.W5(x1 + y1)
        fusion_rates = fusion_rates.squeeze(-1)
        alphas = torch.nn.functional.softmax(fusion_rates, dim=1)
        alphas = alphas.unsqueeze(-1)

        x2 = alphas * x0 + x
        y2 = (1 - alphas) * y0 + y

        fusion = x2 * alphas + y2 * (1 - alphas)

        if dropout:
            fusion = self.__dropout_layer(fusion)

        return fusion  # [batch,hidden]

# TODO:SF-ID Iteration
class IterModel(nn.Module):
    def __init__(self, hid_size, n_slots, n_labels, iteration_num, dropout_rate):
        super(IterModel, self).__init__()
        self.n_labels = n_labels
        self.n_slots = n_slots
        self.V_SF = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V_ID = nn.Linear(in_features=hid_size, out_features=1, bias=None)
        self.W_inte_ans = nn.Linear(in_features=hid_size, out_features=n_labels, bias=None)
        self.W_slot_ans = nn.Linear(in_features=hid_size * 2, out_features=n_slots, bias=None)
        self.__iteration_num = iteration_num
        self.__slot_fusion_layer = SF_FusionLayer(hid_size, dropout_rate, 384)
        self.__intent_fusion_layer = ID_FusionLayer(hid_size, dropout_rate, 384)

    def forward(self, H, c_inte, c_slot_cont, sent_seqs):
        """
        params: h - hidden states [batch_size*len_sents, hid_size]
                c_slot - slot context vector [batch_size*len_sents, hid_size]
                c_inte - intent context vector [batch_size, hid_size]
                sent_seqs-len_sents
        return: intent_output - intent features prepared for softmax [batch_size, n_labels]
                slot_output - slot features prepared for softmax [batch_size, len_sents, n_slots]
        """
        batch_size = c_inte.shape[0]
        output_tensor_list, sent_start_pos = [], 0
        R_inte, R_slot = [], []
        for sent_i in range(len(sent_seqs)):
            sent_end_pos = sent_start_pos + sent_seqs[sent_i]

            # Segment input hidden tensors.
            c_slot = c_slot_cont[sent_start_pos: sent_end_pos, :]  # [seq,hidden]
            c_slot = c_slot.view(1, sent_seqs[sent_i], -1)  # [1,valid_len,hidden]
            h = H[sent_start_pos: sent_end_pos, :].unsqueeze(0)  # [1,valid_len,hidden]
            c_inte1 = c_inte[sent_i].unsqueeze(0)  # [1,hidden]
            r_inte = c_inte1
            len_sents = c_slot.shape[1]  # valid_len
            for iter in range(self.__iteration_num):
                
                r_inte0 = r_inte.unsqueeze(1)
                f_slot = self.__slot_fusion_layer(c_slot, r_inte0)
                f_slot = self.V_SF(f_slot)  # [1, valid_len, hid_size]
                f_slot = torch.sum(f_slot, dim=1)  # [1, hid_size]
                r_slot = f_slot.unsqueeze(1) * c_slot

                
                f_inte = self.__intent_fusion_layer(r_slot, h)
                f_inte = torch.sum(f_inte, dim=1)  # [1, hid_size]
                r_inte = f_inte + c_inte1

            R_inte.append(r_inte)
            R_slot.append(r_slot.squeeze(0))
            sent_start_pos = sent_end_pos

        R_inte = torch.cat(R_inte)  # [batch, hidden]
        R_slot = torch.cat(R_slot, dim=0)  # [batch*valid_len, hidden]

        intent_hiddens = R_inte  # [batch_size, n_intents]
        slot_hiddens = torch.cat((H, R_slot), dim=-1)  # [batch_size, seq_len, 2*hidden_size]
        intent_output = self.W_inte_ans(intent_hiddens)  # [batch_size, n_intents]
        slot_output = self.W_slot_ans(slot_hiddens)  # [batch_size*valid, n_slots]

        return intent_output, slot_output

class EmbeddingCollection(nn.Module):
    """
    TODO: Provide position vector encoding
    Provide word vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Board-casting principle.
        return embedding_x

class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate, bidirectional=True, extra_dim=None):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.__dropout_rate = dropout_rate
        self.__bidirectional = bidirectional
        self.__extra_dim = extra_dim

        lstm_input_dim = self.__embedding_dim + (0 if self.__extra_dim is None else self.__extra_dim)

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.__hidden_dim, batch_first=True,
                                    bidirectional=self.__bidirectional, dropout=self.__dropout_rate, num_layers=1)

    def forward(self, embedded_text, seq_lens, extra_input=None):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([embedded_text, extra_input], dim=-1)
        else:
            input_tensor = embedded_text

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(input_tensor)

        # # Pack and Pad process for input of variable length.
        # packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        # lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        # padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        padded_hiddens, _ = op.pack_and_pad_sequences_for_rnn(dropout_text,
                                                              torch.tensor(seq_lens, device=dropout_text.device),
                                                              self.__lstm_layer)

        # return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)
        return padded_hiddens

class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, slot_fusion_type,
                 embedding_dim=None, extra_input_dim=None, extra_hidden_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_input_dim = extra_input_dim
        self.__extra_hidden_dim = extra_hidden_dim

        # assert self.__hidden_dim == self.__extra_hidden_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(torch.randn(1, self.__embedding_dim), requires_grad=True)

        # Make sure the input dimension of iterative LSTM.
        lstm_input_dim = self.__input_dim + \
                         (0 if self.__extra_input_dim is None else self.__extra_input_dim) + \
                         (0 if self.__embedding_dim is None else self.__embedding_dim)

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.__hidden_dim, batch_first=True,
                                    bidirectional=False, dropout=self.__dropout_rate, num_layers=1)

        self.__slot_fusion_layer = FusionLayer(self.__hidden_dim, self.__hidden_dim,
                                               self.__dropout_rate, slot_fusion_type,
                                               384)

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None,
                extra_hidden=None, attention_module=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """
        if extra_hidden is not None and attention_module is not None:
            extra_hidden_mask = extra_hidden[1]
            extra_hidden = extra_hidden[0]
            assert extra_hidden_mask.shape == extra_hidden.shape[:-1]

        assert extra_hidden is None \
               or (extra_hidden.shape[0] == encoded_hiddens.shape[0] and extra_hidden.shape[-1] == self.__extra_hidden_dim)

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        lstm_output_tensor_list = []
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0) #错位往前移一位
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1) #输入隐藏+foced_tensor
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))

                lstm_out = lstm_out.view(seq_lens[sent_i], -1)
                lstm_output_tensor_list.append(lstm_out)

                if self.__extra_hidden_dim is not None:
                    seg_extra_hidden = extra_hidden[sent_start_pos: sent_end_pos, :]

                    if attention_module is not None:
                        seg_extra_hidden_mask = extra_hidden_mask[sent_start_pos: sent_end_pos]

                        dropout_lstm_out = self.__dropout_layer(lstm_out)
                        dropout_seg_extra_hidden = self.__dropout_layer(seg_extra_hidden)
                        seg_extra_hidden = attention_module(dropout_lstm_out.unsqueeze(1),
                                                            dropout_seg_extra_hidden,
                                                            dropout_seg_extra_hidden,
                                                            mmask=seg_extra_hidden_mask.unsqueeze(1)).squeeze(1)
                linear_out = self.__slot_fusion_layer(lstm_out,
                                                      y=seg_extra_hidden if self.__extra_hidden_dim else None,
                                                      dropout=False)
                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out = lstm_out.view(1, -1)
                    lstm_output_tensor_list.append(lstm_out)

                    if self.__extra_hidden_dim is not None:
                        seg_extra_hidden = extra_hidden[[word_i], :]

                        if attention_module is not None:
                            seg_extra_hidden_mask = extra_hidden_mask[[word_i]]

                            dropout_lstm_out = self.__dropout_layer(lstm_out)
                            dropout_seg_extra_hidden = self.__dropout_layer(seg_extra_hidden)
                            seg_extra_hidden = attention_module(dropout_lstm_out.unsqueeze(1),
                                                                dropout_seg_extra_hidden,
                                                                dropout_seg_extra_hidden,
                                                                mmask=seg_extra_hidden_mask.unsqueeze(1)).squeeze(1)
                    linear_out = self.__slot_fusion_layer(lstm_out,
                                                          y=seg_extra_hidden if self.__extra_hidden_dim else None,
                                                          dropout=False)
                    output_tensor_list.append(linear_out)

                    _, index = linear_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0), torch.cat(lstm_output_tensor_list, dim=0)

class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate, input_linear=True, bilinear=False):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__input_linear = input_linear
        self.__bilinear = bilinear

        # Declare network structures.
        if input_linear and not bilinear:
            self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
            self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
            self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        elif bilinear:
            self.__linear = nn.Linear(self.__query_dim, self.__key_dim)

        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value, mmask=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query) if self.__input_linear and not self.__bilinear else input_query
        linear_key = self.__key_layer(input_key) if self.__input_linear and not self.__bilinear else input_key
        linear_value = self.__value_layer(input_value) if self.__input_linear and not self.__bilinear else input_value

        if self.__input_linear and not self.__bilinear:
            score_tensor = torch.matmul(linear_query, linear_key.transpose(-2, -1)) / math.sqrt(
                self.__hidden_dim if self.__input_linear else self.__query_dim)
        elif self.__bilinear:
            score_tensor = torch.matmul(self.__linear(linear_query), linear_key.transpose(-2, -1))

        if mmask is not None:
            assert mmask.shape == score_tensor.shape
            score_tensor = mmask * score_tensor + (1 - mmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        # forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor

class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, mmask=None):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(dropout_x, dropout_x, dropout_x, mmask=mmask)

        return attention_x

class MLPAttention(nn.Module):

    def __init__(self, input_dim, dropout_rate):
        super(MLPAttention, self).__init__()

        # Record parameters
        self.__input_dim = input_dim
        self.__dropout_rate = dropout_rate

        # Define network structures
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__sent_attention = nn.Linear(self.__input_dim, 1, bias=False)

    def forward(self, encoded_hiddens, rmask=None):
        """
        Merge a sequence of word representations as a sentence representation.
        :param encoded_hiddens: a tensor with shape of [bs, max_len, dim]
        :param rmask: a mask tensor with shape of [bs, max_len]
        :return:
        """
        # TODO: Do dropout ?
        dropout_input = self.__dropout_layer(encoded_hiddens)
        score_tensor = self.__sent_attention(dropout_input).squeeze(-1)

        if rmask is not None:
            assert score_tensor.shape == rmask.shape, "{} vs {}".format(score_tensor.shape, rmask.shape)
            score_tensor = rmask * score_tensor + (1 - rmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        # matrix multiplication: [bs, 1, max_len] * [bs, max_len, dim] => [bs, 1, dim]
        sent_output = torch.matmul(score_tensor.unsqueeze(1), dropout_input).squeeze(1)

        return sent_output #[batch,dim]

