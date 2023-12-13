# -*- coding: utf-8 -*-#
"""
@CreateTime :       2022/12/28 21:25
@Author     :       Qingpeng Wen
@File       :       config.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2022/12/28 23:35
"""

import argparse

parser = argparse.ArgumentParser()

'''Training and evaluation choose.'''
parser.add_argument('--do_evaluation', '-eval', action="store_true", default=True)

'''the dataset dictionary'''
parser.add_argument('--data_dir', '-dd', type=str, default='esac/code/data/cais')
# parser.add_argument('--data_dir', '-dd', type=str, default='esac/code/data/ecdt')
parser.add_argument('--train_file_name', '-train_file', type=str, default='train.txt')
parser.add_argument('--valid_file_name', '-valid_file', type=str, default='dev.txt')
parser.add_argument('--test_file_name', '-test_file', type=str, default='test.txt')

'''For training and evaluation saves'''
parser.add_argument('--save_dir', '-sd', type=str, default='save_cais')
# parser.add_argument('--save_dir', '-sd', type=str, default='save_ecdt')

'''Training parameters.'''
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=300)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument("--max_grad_norm", "-mgn", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.5)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

'''model parameters.'''
parser.add_argument('--percent_of_encoder_hidden_dim', '-pehd', type=float, default=0.5)
parser.add_argument('--single_channel_intent', '-sci', action="store_true", default=False)
parser.add_argument('--single_channel_slot', '-scs', action="store_true", default=False)
parser.add_argument('--char_embedding_dim', '-ced', type=int, default=64)
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--char_attention_hidden_dim', '-cahd', type=int, default=1024)
parser.add_argument('--word_attention_hidden_dim', '-wahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
parser.add_argument('--intent_fusion_type', '-ift', type=str, default='bilinear')  
parser.add_argument('--slot_fusion_type', '-sft', type=str, default='bilinear')  
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=16)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--undirectional_word_level_slot_encoder', '-udwse', action="store_true", default=False)
