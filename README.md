cemotion 是 Python 下的中文 NLP 库，可以进行中文情感倾向分析、通用领域中文分词。

cemotion 2.0 模型使用 BERT (Bidirectional Encoder Representations from Transformers) 训练得到，会为中文文本返回 0～1 之间的情感倾向置信度 (情感极性 0 为消极，1 为积极)。

此外，新加入的Cegementor中文分词类采用BAStructBERT通用领域中文分词模型对文本按语义进行分词。

使用 cemotion，您将能够：
- 批量分析中文文本的情感
- 批量进行中文文本分词
- 部署至 Linux、macOS、Windows 等生产环境中 (支持 Apple Silicon)

该模块依赖于 PyTorch 环境（会自动安装），要求 Python 3.8 或更高版本，较老的机器可能无法运行。

注意，cemotion 会自动调用 NVIDIA 和 Apple Silicon 的 GPU。如果没有 GPU，则使用 CPU 推理。

### 安装方法

1.进入命令窗口，创建虚拟环境，依次输入以下命令

Linux 和 macOS:

```bash
python3 -m venv venv #创建虚拟环境
. venv/bin/activate #激活虚拟环境
```

Windows:

```bash
python -m venv venv #创建虚拟环境
venv\Scripts\activate #激活虚拟环境
```

2.安装cemotion库，依次输入

```bash
pip install --upgrade pip
pip install cemotion
```

### 链接

- GitHub https://github.com/Cyberbolt/Cemotion
- 电光笔记 https://www.cyberlight.xyz/


### 使用方法
```python
#按文本字符串分析
from cemotion import Cemotion

str_text1 = '配置顶级，不解释，手机需要的各个方面都很完美'
str_text2 = '院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！'

c = Cemotion()
print(f'"{str_text1}"\n预测值:{c.predict(str_text1)}', '\n')
print(f'"{str_text2}"\n预测值:{c.predict(str_text2)}' , '\n')
```


```
#返回内容(该模块返回了这句话的情感置信度，值在0到1之间):
" 配置顶级，不解释，手机需要的各个方面都很完美 "
 预测值:0.999962 

" 院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！ "
 预测值:0.000147
```




```python
#使用列表进行批量分析
from cemotion import Cemotion

list_text = ['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！',
'总而言之，是一家不会再去的店。']

c = Cemotion()
print(c.predict(list_text))
```


```
#返回内容(该模块返回了列表中每句话的情感置信度，值在0到1之间):
[['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 0.999962], ['总而言之，是一家不会再去的店。', 0.000194]]
```



```
#单个文本进行分词
from cemotion import Cegmentor

text = '这辆车的内饰设计非常现代，而且用料考究，给人一种豪华的感觉。'

model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'
segmenter = Cegmentor(model_id)
segmentation_result = segmenter.segment(text)
print(segmentation_result)
```

```
#返回内容为单句分词
['这', '辆', '车', '的', '内饰', '设计', '非常', '现代', '，', '而且', '用料', '考究', '，', '给', '人', '一', '种', '豪华', '的', '感觉', '。']
```



```
from cemotion import Cegmentor

text = '这辆车的内饰设计非常现代，而且用料考究，给人一种豪华的感觉。'

list_text = [
    '随着科技的发展，智能手机的功能越来越强大，给我们的生活带来了很多便利。',
    '他从小就对天文学充满好奇，立志要成为一名宇航员，探索宇宙的奥秘。',
    '这种新型的太阳能电池板转换效率高，而且环保，有望在未来得到广泛应用。'
]
model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'
segmenter = Cegmentor(model_id)
segmentation_result = segmenter.segment(list_text)
print(segmentation_result)
```

```
#返回内容为分词列表
[
['随着', '科技', '的', '发展', '，', '智能', '手机', '的', '功能', '越来越', '强大', '，', '给', '我们', '的', '生活', '带来', '了', '很多', '便利', '。'],
['他', '从小', '就', '对', '天文学', '充满', '好奇', '，', '立志', '要', '成为', '一', '名', '宇航员', '，', '探索', '宇宙', '的', '奥秘', '。'],
['这种', '新型', '的', '太阳能', '电池板', '转换', '效率', '高', '，', '而且', '环保', '，', '有望', '在', '未来', '得到', '广泛', '应用', '。']
]
```



### 2.0 版本主要更新内容

1.替换依赖 TensorFlow 为 PyTorch

2.将老版本的 BRNN + LSTM 更改为 BERT 模型

此外，2.0 版本的接口和老版本保持相同，您可以无缝切换。
