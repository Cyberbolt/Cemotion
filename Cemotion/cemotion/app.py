'''
    本类用于 情感倾向分析
    预测值为大小0～1之间的置信度
'''

import os
from sys import implementation

import numpy as np
import tensorflow as tf

from cemotion.dataset import DataSet
from cemotion.download import download_from_url


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #只显示error和warining信息


#检测所需文件是否存在，不存在则下载
def check_env(url, path):
    if os.path.exists(path):
        return
    else:
        print('Downloading the required environment, Please wait.\
              \nIf you are using China Telecom, you may only get faster download speeds during the day.')
        download_from_url(url, path)


class Cemotion:
    def __init__(self):
        current_path = os.path.dirname(__file__) #当前模块的路径
        #保存模型的路径
        model_path = current_path + '/models/rnn_emotion_x86_1.0.h5'
        #保存中文词典路径
        dictionary_path = current_path + '/models/requirements/big_Chinese_Words_Map.dict'        
        #检测所需文件是否存在，判断是否下载
        #若主链接无法使用，使用备用链接
        # try:
        #     check_env('https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaVdOR2ZlUEx6NWszUVJVdGhaQWZJNVROVTlSP2U9b2pJOEtF.dict', 
        #           dictionary_path)
        # except:
        check_env('https://www.cyberlight.xyz/static/file/cemotion/big_Chinese_Words_Map.dict', 
                dictionary_path)
        #若主链接无法使用，使用备用链接
        # try:
        #     check_env('https://onedrive.gimhoy.com/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaVdOR2ZlUEx6NWszVndYZ2I3SmJQdDR6b0pqP2U9UjlzZmRR.h5', 
        #               model_path)
        # except:
        check_env('https://www.cyberlight.xyz/static/file/cemotion/rnn_emotion_x86_1.0.h5', 
                    model_path)
        #加载rnn模型
        self.__rnn = tf.keras.models.load_model(model_path)
        #加载数据集实例
        self.__dataset = DataSet(400, dictionary_path) #句子最大长度 #字典路径      
        
    def predict(self, text):
        #输入内容为文字时 返回 正负概率
        if type(text) == type('text mode'):
            # print('text mode')
            list_text = [text] #将文本转为列表
            #获取预测值  预测一个值时使用predict_on_batch
            prediction = self.__rnn.predict_on_batch(self.__dataset.data_to_train(list_text))[0][0]
            
            return round(prediction, 6)
        
        #输入列容为列表时 返回 带正负概率的列表
        elif type(text) == type(['list mode']) or type(text) == type(np.array(['list mode'])):
            #如果是numpy数组 则 转为列表
            if type(text) == type(np.array(['list mode'])):
                text = text.tolist()
            
            # print('list mode')
            list_text = text
            prediction = self.__rnn.predict(self.__dataset.data_to_train(list_text))
            
            list_new = [] #第一列保存文字，第二列保存文字对应的情感
            #生成新表
            for one, two in zip(list_text, prediction):
                list_new.append([one, round(two[0], 6)])
                
            return list_new