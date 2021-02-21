Cemotion是Python下的中文NLP库，可以进行 中文情感倾向分析。

Cemotion的模型经循环神经网络训练得到，会为 中文文本 返回 0～1之间的 情感倾向置信度。您可以批量分析中文文本的情感，并部署至Linux、Mac OS、Windows等生产环境中，无需关注内部原理。

该模块依赖于TensorFlow环境（会自动安装），较老的机器可能无法运行。



### 安装方法

1.进入命令窗口，创建虚拟环境，依次输入以下命令

Linux和Mac OS:

```
python3 -m venv venv #创建虚拟环境
. venv/bin/activate #激活虚拟环境
```

附:Apple Silicon安装方法

Apple Silicon请参考 [https://pypi.org/project/Cemotion-apple/](https://pypi.org/project/Cemotion-apple/) 此文档安装


Windows:

```
python -m venv venv #创建虚拟环境
venv\Scripts\activate #激活虚拟环境
```

2.安装cemotion库，依次输入

```
pip install --upgrade pip
pip install cemotion
```




### 使用方法
```
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
text mode
" 配置顶级，不解释，手机需要的各个方面都很完美 "
 预测值:0.999931 

text mode
" 院线看电影这么多年以来，这是我第一次看电影睡着了。简直是史上最大烂片！没有之一！侮辱智商！大家小心警惕！千万不要上当！再也不要看了！ "
 预测值:0.000001 
```




```
#使用列表进行批量分析
from cemotion import Cemotion

list_text = ['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！',
'总而言之，是一家不会再去的店。']

c = Cemotion()
print(c.predict(list_text))
```


```
#返回内容(该模块返回了列表中每句话的情感置信度，值在0到1之间):
list mode
[['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 0.999907], ['总而言之，是一家不会再去的店。', 0.049015]]
```