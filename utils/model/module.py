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

from utils import operation as op
from utils.model.layers import FusionLayer, EmbeddingCollection, LSTMEncoder, LSTMDecoder, SelfAttention, MLPAttention, IterModel
from utils.data_loader.functions import MODEL_PATH_MAP

MASK_VALUE = -2 ** 32 + 1

class ModelManager(nn.Module):

    def __init__(self, args, num_char, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        # hyper-parameters
        self.__num_char = num_char
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an char embedding object.
        self.__char_embedding = EmbeddingCollection(self.__num_char, self.__args.char_embedding_dim)
        # Initialize an word embedding object.
        self.__word_embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim)

        # Initialize an LSTM Encoder object for char level
        self.__char_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for char level
        self.__char_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.char_attention_hidden_dim,
                                                  self.__args.attention_output_dim, self.__args.dropout_rate)

        # Initialize an LSTM Encoder object for word level
        self.__word_encoder = LSTMEncoder(self.__args.word_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for word level
        self.__word_attention = SelfAttention(self.__args.word_embedding_dim, self.__args.word_attention_hidden_dim,
                                              self.__args.attention_output_dim, self.__args.dropout_rate)

        self.__encoder_output_dim = self.__args.encoder_hidden_dim + self.__args.attention_output_dim

        # dropout layer
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)

        # TODO: MLP Attention
        self.__char_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)
        self.__word_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)

        # TODO: Fusion
        self.__intent_fusion_layer = FusionLayer(self.__encoder_output_dim, self.__encoder_output_dim,
                                                 self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                 384)
        self.__slot_hidden_fusion_layer = FusionLayer(self.__encoder_output_dim, self.__encoder_output_dim,
                                                      self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                      384)
        self.__hidden_fusion_layer = FusionLayer(self.__encoder_output_dim, self.__encoder_output_dim,
                                                 self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                 384)

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__args.intent_embedding_dim)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent, self.__args.intent_embedding_dim)

        # Initialize an Encoder and Decoder object for char-level slot.
        self.__decoder = LSTMDecoder(
            self.__encoder_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate, self.__args.slot_fusion_type,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_input_dim=self.__args.intent_embedding_dim,
            extra_hidden_dim=self.__args.slot_decoder_hidden_dim
        )
        
        self.__word_slot_encoder = LSTMEncoder(self.__encoder_output_dim, self.__encoder_output_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=True
                                               )
        self.__char_slot_encoder = LSTMEncoder(self.__encoder_output_dim, self.__encoder_output_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=True
                                               )
        self.__iterModel = IterModel(self.__encoder_output_dim, self.__num_slot, self.__num_intent, 4,
                                     self.__args.dropout_rate)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """
        print('Model parameters are listed as follows:\n')

        print('\tdropout rate:						                    {};'.format(self.__args.dropout_rate))
        print('\tnumber of char:						                {};'.format(self.__num_char))
        print('\tnumber of word:                                        {};'.format(self.__num_word))
        print('\tnumber of slot:                                        {};'.format(self.__num_slot))
        print('\tnumber of intent:						                {};'.format(self.__num_intent))
        print('\tchar embedding dimension:				                {};'.format(self.__args.char_embedding_dim))
        print('\tword embedding dimension:				                {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				                {};'.format(self.__args.encoder_hidden_dim))
        print('\thidden dimension of char-level self-attention:         {};'.format(self.__args.char_attention_hidden_dim))
        print('\thidden dimension of word-level self-attention:         {};'.format(self.__args.word_attention_hidden_dim))
        print('\toutput dimension of self-attention:                    {};'.format(self.__args.attention_output_dim))
        print('\tintent fusion type:                                    {};'.format(self.__args.intent_fusion_type))
        print('\tslot fusion type:                                      {};'.format(self.__args.slot_fusion_type))
        print('\tdimension of slot embedding:			                {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of intent embedding:			                {};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot decoder hidden:  	                {};'.format(self.__args.slot_decoder_hidden_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, char_text, char_seq_lens, word_text, word_seq_lens, align_info,
                n_predicts=None, forced_slot=None):
        """
        :param char_text: list of list of char ids
        :param char_seq_lens: list of the number of chars, e.g. [6, 7, 7]
        :param word_text: list of list of word ids
        :param word_seq_lens: list of the number of words, e.g. [4, 3, 4]
        :param align_info: list of list of the number of chars in each word, e.g. [ [1, 2, 1, 2], [2, 2, 3], [2, 1, 3, 1] ]
        """


        char_tensor = self.__char_embedding(char_text)
        word_tensor = self.__word_embedding(word_text)

        # Get mask
        device = word_tensor.device
        char_rmask, char_mmask = op.generate_mask(char_seq_lens, device)
        word_rmask, word_mmask = op.generate_mask(word_seq_lens, device)

        # Pass char encoder
        char_lstm_hiddens = self.__char_encoder(char_tensor, char_seq_lens)  # [batch,seq,hidden]
        char_attention_hiddens = self.__char_attention(char_tensor, mmask=char_mmask)  # [batch,seq,hidden]
        char_hiddens = torch.cat([char_attention_hiddens, char_lstm_hiddens], dim=-1)  # [batch,seq,hidden1+hidden2]

        # Pass word encoder
        word_lstm_hiddens = self.__word_encoder(word_tensor, word_seq_lens)
        word_attention_hiddens = self.__word_attention(word_tensor,  mmask=word_mmask)
        word_hiddens = torch.cat([word_attention_hiddens, word_lstm_hiddens], dim=-1)  # [batch,seq,hidden]

        # MLP Attention
        word_sent_output = self.__word_sent_attention(word_hiddens, rmask=word_rmask)  # [batch,dim]
        char_sent_output = self.__char_sent_attention(char_hiddens, rmask=char_rmask)  # [batch,dim]

        # Intent Layer
        intent_hidden = self.__intent_fusion_layer(char_sent_output, word_sent_output)  # [batch,hidden]
        word_slot_out = self.__word_slot_encoder(word_hiddens, word_seq_lens)  # [batch,seq,hid]

        # word_seq_lens:[batch,valid_len]
        char_slot_out = self.__char_slot_encoder(char_hiddens,char_seq_lens) #[batch,seq,hid]
        flat_word_slot_out = torch.cat([word_slot_out[i][:word_seq_lens[i]]
                                               for i in range(0, len(word_seq_lens))], dim=0) #[batch*valid_len,hid]
        aligned_word_slot_out = op.char_word_alignment(flat_word_slot_out, char_seq_lens, word_seq_lens, align_info)#??
        flat_char_slot_out = torch.cat([char_slot_out[i][:char_seq_lens[i]]
                                               for i in range(0, len(char_seq_lens))], dim=0) #[batch*valid_len,hid]
        slot_out = self.__slot_hidden_fusion_layer(flat_char_slot_out, aligned_word_slot_out) #[]

        flat_char_hidden = torch.cat([char_hiddens[i][:char_seq_lens[i]]
                                        for i in range(0, len(char_seq_lens))], dim=0)  # [batch*valid_len,hid]
        flat_word_hidden = torch.cat([word_hiddens[i][:word_seq_lens[i]]
                                      for i in range(0, len(word_seq_lens))], dim=0)  #
        aligned_word_hidden = op.char_word_alignment(flat_word_hidden, char_seq_lens, word_seq_lens,
                                                       align_info) # [batch*valid_len,hid]
        hiddens = self.__hidden_fusion_layer(flat_char_hidden, aligned_word_hidden)

        pred_intent, pred_slot = self.__iterModel(hiddens, intent_hidden, slot_out, char_seq_lens)

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), \
                   intent_index.cpu().data.numpy().tolist()