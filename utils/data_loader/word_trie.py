# -*- coding: utf-8 -*-
"""
@CreateTime :       2022/12/28 21:25
@Author     :       QingPeng Wen
@File       :       word_trie.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2022/12/28 23:35
"""
_end = "_end_"


class Word_Trie:
    def __init__(self):
        self.root = dict()

    def recursive_search(self, word_list, allow_single_char=False):
        match_list = []
        min_word_len = 0 if allow_single_char else 1
        while len(word_list) > min_word_len:
            if self.search(word_list):
                match_list.append("".join(word_list))
            del word_list[-1]
        return match_list

    def search(self, word):
        current_dict = self.root
        for char in word:
            if char in current_dict:
                current_dict = current_dict[char]
            else:
                return False
        else:
            if _end in current_dict:
                return True
            else:
                return False

    def insert(self, word):
        if isinstance(word, (list, tuple)):
            for element in word:
                self.insert(element)
            return

        assert isinstance(word, str)

        current_dict = self.root
        for char in word:
            current_dict = current_dict.setdefault(char, {})
        current_dict[_end] = _end
