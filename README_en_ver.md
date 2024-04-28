Cemotion is a Chinese NLP (Natural Language Processing) library for Python that can perform sentiment analysis and general domain Chinese word segmentation.

The Cemotion 2.0 model is trained using BERT (Bidirectional Encoder Representations from Transformers), which returns a sentiment confidence score between 0 and 1 for Chinese text (sentiment polarity with 0 being negative and 1 being positive).

Additionally, the newly added Cegementor Chinese word segmentation class uses the BAStructBERT general domain Chinese word segmentation model to segment text semantically.

With Cemotion, you will be able to:

- Analyze the sentiment of Chinese text in batches
- Perform Chinese text segmentation in batches
- Deploy to production environments such as Linux, macOS, Windows (supports Apple Silicon)

This module depends on the PyTorch environment (which will be installed automatically) and requires Python version 3.8 or higher. Older machines may not be able to run it.

Please note that Cemotion will automatically utilize the GPUs of NVIDIA and Apple Silicon. If there is no GPU available, it will use CPU for inference.

### Installation Method

1. Enter the command window and create a virtual environment by entering the following commands in sequence.

For Linux and macOS:

```bash
python3 -m venv venv # Create a virtual environment
. venv/bin/activate # Activate the virtual environment
```

For Windows:

```bash
python -m venv venv # Create a virtual environment
venv\Scripts\activate # Activate the virtual environment
```

2. Install the Cemotion library by entering the following commands in sequence:

```bash
pip install --upgrade pip
pip install cemotion
```

### Links

- GitHub: https://github.com/Cyberbolt/Cemotion
- Cyberlight Notes: https://www.cyberlight.xyz/

### Usage

#### Chinese Sentiment Classification

```python
# Analyze by text string
from cemotion import Cemotion

str_text1 = '配置顶级，不解释，手机需要的各个方面都很完美'
str_text2 = '院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！'

c = Cemotion()
print(f'"{str_text1}"\n预测值:{c.predict(str_text1)}', '\n')
print(f'"{str_text2}"\n预测值:{c.predict(str_text2)}' , '\n')
```


```
# Return Content (This module returns the sentiment confidence score for the sentence, with a value ranging from 0 to 1):
" 配置顶级，不解释，手机需要的各个方面都很完美 "
 预测值:0.999962 

" 院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！ "
 预测值:0.000147
```




```python
# Using a list for batch analysis
from cemotion import Cemotion

list_text = ['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！',
'总而言之，是一家不会再去的店。']

c = Cemotion()
print(c.predict(list_text))
```


```
# Return Content (This module returns the sentiment confidence score for the sentence, with a value ranging from 0 to 1).:
[['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 0.999962], ['总而言之，是一家不会再去的店。', 0.000194]]
```



#### Chinese text segmentation

```python
# Segmentation of a single text
from cemotion import Cegmentor

text = '这辆车的内饰设计非常现代，而且用料考究，给人一种豪华的感觉。'

model_id = 'damo/nlp_structbert_word-segmentation_chinese-base'
segmenter = Cegmentor(model_id)
segmentation_result = segmenter.segment(text)
print(segmentation_result)
```

```
# Return content as a single sentence segmentation
['这', '辆', '车', '的', '内饰', '设计', '非常', '现代', '，', '而且', '用料', '考究', '，', '给', '人', '一', '种', '豪华', '的', '感觉', '。']
```



```python
# Using a list for batch Chinese text segmentation
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
# Return content as a list of segmentations
[
['随着', '科技', '的', '发展', '，', '智能', '手机', '的', '功能', '越来越', '强大', '，', '给', '我们', '的', '生活', '带来', '了', '很多', '便利', '。'],
['他', '从小', '就', '对', '天文学', '充满', '好奇', '，', '立志', '要', '成为', '一', '名', '宇航员', '，', '探索', '宇宙', '的', '奥秘', '。'],
['这种', '新型', '的', '太阳能', '电池板', '转换', '效率', '高', '，', '而且', '环保', '，', '有望', '在', '未来', '得到', '广泛', '应用', '。']
]
```



### Main Updates in Version 2.0

1. Replaced the dependency on TensorFlow with PyTorch.

2. Changed the old version's BRNN + LSTM to the BERT model.

Additionally, the interface of version 2.0 remains the same as the old version, allowing for seamless switching.
