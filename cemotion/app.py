'''
    本类用于 情感倾向分析
    预测值为大小0～1之间的置信度
'''

import os
import logging as log
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, logging
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor
from cemotion.download import download_from_url
import warnings

# 全局忽略 UserWarning 类型的警告
warnings.simplefilter(action='ignore', category=UserWarning)
# 全局忽略 FutureWarning 类型的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

log.getLogger('modelscope').setLevel(log.CRITICAL)
logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 检测所需文件是否存在，不存在则下载
def check_env(url):
    if not os.path.exists('.cemotion_cache'):
        os.mkdir('.cemotion_cache')
    if os.path.exists('.cemotion_cache/cemotion_2.0.pt'):
        return
    else:
        print('Downloading the required environment, Please wait.\
              \nIf you are using China Telecom, you may only get faster download speeds during the day.')
        download_from_url(url, '.cemotion_cache/cemotion_2.0.pt')


# 定义模型
class SentimentClassifier(torch.nn.Module):

    def __init__(self, num_classes=1):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese', num_labels=num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs[0]


# 定义从本地读取模型的函数
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=torch.device(device)),strict=False)
    return model


class Cemotion:
    def __init__(self):
        #检测所需文件是否存在，判断是否下载
        check_env('https://github.com/Cyberbolt/Cemotion/releases/download/2.0/cemotion_2.0.pt')

        # 检测 GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # 加载模型
        model = SentimentClassifier(num_classes=1)
        self.model = load_model(model, '.cemotion_cache/cemotion_2.0.pt', device)
        self.model.to(device)
        self.device = device

    def predict(self, text):
        # 输入内容为文字时 返回 正负概率
        if type(text) == type('text mode'):
            list_text = [text]  # 将文本转为列表
            prediction = self.deal(list_text)[0]
            return round(prediction, 6)

        # 输入列容为列表时 返回 带正负概率的列表
        elif type(text) == type(['list mode']) or type(text) == type(np.array(['list mode'])):
            # 如果是numpy数组 则 转为列表
            if type(text) == type(np.array(['list mode'])):
                text = text.tolist()

            list_text = text
            prediction = self.deal(list_text)
            list_new = []  # 第一列保存文字，第二列保存文字对应的情感
            # 生成新表
            for one, two in zip(list_text, prediction):
                list_new.append([one, round(two, 6)])

            return list_new

    def deal(self, list_text: list) -> list:
        '''
            封装预测功能
        '''
        predictions = []
        for sentence in list_text:
            inputs = tokenizer.encode_plus(
                sentence, add_special_tokens=True, max_length=128, padding='max_length', truncation=True)
            input_ids = torch.tensor(
                inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(
                inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(
                inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).squeeze(1)
            predict = torch.sigmoid(outputs).cpu().detach().numpy()[0]
            predictions.append(predict)
        return predictions
    




class Cegmentor:
    def __init__(self, model_id):
        # 加载模型和分词器
        self.model = Model.from_pretrained(model_id)
        self.tokenizer = TokenClassificationTransformersPreprocessor(self.model.model_dir)
        self.pipeline = pipeline(task=Tasks.token_classification, model=self.model, preprocessor=self.tokenizer)

    def segment(self, text):
        # 检查输入是否为列表
        if isinstance(text, list):
            # 对列表中的每个文本执行分词，并返回分词结果的列表
            return [self._segment_single(t) for t in text]
        else:
            # 执行单个文本的分词
            return self._segment_single(text)

    def _segment_single(self, text):
        # 使用pipeline进行分词
        result = self.pipeline(input=text)
        
        # 只提取分词结果
        words = [output['span'] for output in result['output']]
        
        # 返回分词结果
        return words

    def tag(self, text):
        # 使用pipeline进行分词和词性标注
        result = self.pipeline(input=text)
        # 只提取分词结果和词性标注
        tags = [output['type'] for output in result['output']]
        words = [output['span'] for output in result['output']]
        
        # 返回分词结果和词性的元组列表
        return list(zip(words, tags))