'''
此类用于数据转换
将 普通数据 转为模型能读懂的数据
'''

import jieba
import logging
import joblib
import numpy as np


jieba.setLogLevel(logging.INFO) #不输出jieba日志


class DataSet:
    def __init__(self, sentence_len, dictionary_path):
        #加载中文词典
        self.__Chinese_dict = joblib.load(dictionary_path) 
        self.sentence_len = int(sentence_len) #句子长度 单位:词
    
    #将每个句子的列表 转为 编码后的列表
    def data_to_train(self, list_str):
        Chinese_dict = self.__Chinese_dict #提取中文词典
        Chinese_reverse = {v:k for k,v in Chinese_dict.items()} #反转字典
        
        #将 句子表 编码
        list_coding = [] #保存编码的整个表
        #遍历每个句子
        for row in list_str:
            text_coding = [] #保存编码的句子
            
            #分词
            sentence = list(jieba.cut(row, cut_all=False) )
            #遍历句子中每个单词 #只保存词库有的单词
            for one in sentence:
                try:
                    text_coding.append(Chinese_reverse[one] ) #通过反转的字典查每个单词的索引
                except:
                    continue
            
            lenth = len(text_coding)
            #如果句子大于698个词，只取前698个，否则补齐698个
            if lenth > self.sentence_len:
                text_coding = text_coding[:self.sentence_len]
            elif lenth < self.sentence_len:
                text_coding.extend([0 for i in range(self.sentence_len-lenth)])
        
            list_coding.append(text_coding) #将编码后的句子保存到表
        
        return np.array(list_coding)