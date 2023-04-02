Cemotion 是 Python 下的中文 NLP 库，可以进行中文情感倾向分析。

Cemotion 2.0 模型使用 BERT (Bidirectional Encoder Representations from Transformers) 训练得到，会为中文文本返回 0～1 之间的情感倾向置信度 (情感极性 0 为消极，1 为积极)。

使用 Cemotion，您将能够：
- 批量分析中文文本的情感
- 部署至 Linux、macOS、Windows 等生产环境中 (支持 Apple Silicon)

该模块依赖于 PyTorch 环境（会自动安装），要求 Python 3.8 或更高版本，较老的机器可能无法运行。

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

2.安装Cemotion库，依次输入

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
print('"', str_text1 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text1) ) , '\n')
print('"', str_text2 , '"\n' , '预测值:{:6f}'.format(c.predict(str_text2) ) , '\n')
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

### 2.0 版本主要更新内容

1.替换依赖 TensorFlow 为 PyTorch

2.将老版本的 BRNN + LSTM 更改为 BERT 模型

此外，2.0 版本的接口和老版本保持相同，您可以无缝切换。
