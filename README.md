# Cemotion
Cemotion是Python下的中文NLP库，可以进行 中文情感倾向分析。
Cemotion的模型经循环神经网络训练得到，会为 中文文本 返回 0～1之间的 情感倾向置信度。您可以批量分析中文文本的情感，并部署至Linux、Mac OS、Windows等生产环境中，无需关注内部原理。

安装方法
1.进入命令窗口，创建虚拟环境，依次输入以下命令
Linux和Mac OS:
python3 -m venv venv #创建虚拟环境
. venv/bin/activate #激活虚拟环境
Windows:
python -m venv venv #创建虚拟环境
venv\Scripts\activate #激活虚拟环境
2.安装cemotion库，依次输入
pip install --upgrade pip
pip install cemotion


使用方法

#按文本字符串分析
from cemotion import Cemotion
str_text = '内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！'
c = Cemotion()
print(c.predict(str_text))
----------
返回内容:
text mode
0.7465
----------

#使用列表进行批量分析
from cemotion import Cemotion
list_text = ['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 
            '总而言之，是一家不会再去的店。']
c = Cemotion()
print(c.predict(list_text))
----------
返回内容:
list mode
[['内饰蛮年轻的，而且看上去质感都蛮好，貌似本田所有车都有点相似，满高档的！', 0.7465], ['总而言之，是一家不会再去的店。', 0.7457]]
----------

