Cemotion是Python下的中文NLP库，可以进行 中文情感倾向分析。

Cemotion的模型经循环神经网络训练得到，会为 中文文本 返回 0～1之间的 情感倾向置信度。您可以批量分析中文文本的情感，并部署至Linux、Mac OS、Windows等生产环境中，无需关注内部原理。

该模块供Apple Silicon使用，已经过M1测试。请按该文档安装ARM Python、TensorFlow、scikit-learn环境。



### 安装方法

前提:
根据 [https://www.cyberlight.xyz/passage/tensorflow-apple-m1](url) 此文方法安装ARM Python和TensorFlow（TensorFlow需要装到conda虚拟环境中，通读全文后，请使用文章末尾的方法安装TensorFlow）

此时，我们假定您已安装相关环境，并创建了名为py38的conda虚拟环境

1.进入命令窗口，激活conda虚拟环境，安装scikit-learn


```
conda activate py38 #激活虚拟环境 此处虚拟环境名称为py38（您可以自定义名称）
conda install scikit-learn #安装scikit-learn
```


之后输入以下命令安装cemotion
```
pip install --upgrade pip
pip install cemotion-apple
```



### 使用方法
```
#按文本字符串分析
from cemotion import Cemotion
str_text = '内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！'
c = Cemotion()
print(c.predict(str_text))
```


```
返回内容:
text mode
0.7465
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
返回内容:
list mode
[['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 0.7465], ['总而言之，是一家不会再去的店。', 0.7457]]
```

