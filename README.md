# Net-class-hand-acid-acid
paddlehub ocr实践
# 网课手酸酸，眼花花，救星来啦！ 
paddlehubocr实践

### 网课看完笔油见底，两手酸酸，头皮麻麻，本子满满密密麻麻。
### 你还在为网课抄不完的笔记烦恼嘛？
### 你还在为网课那弄不完的无底洞头疼嘛?
## 你们的救星来了！！！！

### 这里是大家的三岁，万能小白三岁，大家的苦恼网课笔记手麻麻，用paddleOCR解决啦！！！

### 让我们一起来看看叭！
#### 有关的环境配置（项目基于 paddle1.8.0）
该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper


```python
## 配置有关的环境
!hub install chinese_ocr_db_crnn_mobile==1.1.0
!pip install shapely
!pip install pyclipper
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Module chinese_ocr_db_crnn_mobile-1.1.0 already installed in /home/aistudio/.paddlehub/modules/chinese_ocr_db_crnn_mobile
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: shapely in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.7.1)
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: pyclipper in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.2.0)


本次使用的OCR模型是haddlehubOCR chinese_ocr_db_crnn_server的1.1.0版本。
## 模型概述
chinese_ocr_db_crnn_server Module用于识别图片当中的汉字。其基于chinese_text_detection_db_server检测得到的文本框，继续识别文本框中的中文文字。之后对检测文本框进行角度分类。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个通用的OCR模型，支持直接预测。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201113164728186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyMzA5Mw==,size_16,color_FFFFFF,t_70#pic_center)


### 文件处理


```python
### 判定有无规定文件夹
import os #导入os库


path = './images/';  # 将视频切割为图片存放的路径


if os.path.exists(path):
    print('文件夹已经存在')
else:
    os.mkdir(path)  #不存在即创建
    print('创建成功')

if os.path.exists(path):
    print('文件已经存在')
else:
    os.mknod(txtpath)  #不存在即创建
    print('文件创建成功')
```

    文件夹已经存在
    文件已经存在


### 视频分割


```python
### 把视频按照一定的帧进行切割生成照片并存储到指定文件
import cv2

vc = cv2.VideoCapture(r'./ceshs.mp4')  # 读入视频文件，命名cv
n = 1  # 计数

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
 
timeF = 15  # 视频帧计数间隔频率
 
i = 0
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    try:
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            cv2.imwrite(r''+path+'{}.jpg'.format(i), frame)  # 存储为图像
        n = n + 1
        cv2.waitKey(1)
    except:
        break
vc.release()
```


```python
# !hub run chinese_ocr_db_crnn_mobile --input_path "1596726259(1).jpg" # 使用Linux环境使用OCR
```


```python
### 把有关的文件进行遍历后使用OCR进行生成文字。
import paddlehub as hub
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
data = os.listdir(path) #读取保存图片的文件夹

for photo in data:
    print(photo)
    lena = mpimg.imread(f'./images/{photo}')
    plt.figure(figsize=(10,10))
    # lena.shape
    plt.imshow(lena) 
    plt.axis('off') 
    plt.show()
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    try:
        result = ocr.recognize_text(images=[cv2.imread(f'./images/{photo}')])
    except:
        pass
    # result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    for text in result[0]['data']:
        print(text['text'])
```

    9.jpg



![在这里插入图片描述](https://img-blog.csdnimg.cn/20201113170443413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyMzA5Mw==,size_16,color_FFFFFF,t_70#pic_center)


    [32m[2020-10-30 17:55:33,430] [    INFO] - Installing chinese_ocr_db_crnn_mobile module[0m
    [32m[2020-10-30 17:55:33,616] [    INFO] - Module chinese_ocr_db_crnn_mobile already installed in /home/aistudio/.paddlehub/modules/chinese_ocr_db_crnn_mobile[0m
    [32m[2020-10-30 17:55:33,927] [    INFO] - Installing chinese_text_detection_db_mobile module-1.0.3[0m
    [32m[2020-10-30 17:55:33,929] [    INFO] - Module chinese_text_detection_db_mobile-1.0.3 already installed in /home/aistudio/.paddlehub/modules/chinese_text_detection_db_mobile[0m


    人工智能无处不在
    1/飞桨
    搜素引擎：网页、图片、视频、新闻、学术、地图
    信息推荐：新闻、商品、游戏、书籍
    智能助理
    图片识别：人像、用品、动物、交通工具
    用户分析：社交网络、影评、商品评论
    智能图像
    理解
    机器翻译、摘要生成.
    逻辑操作符
    描述
    机器翻译
    and
    网络购物
    如果两个操作数中的任何一个为True则eondition变为True
    期
    智能化推荐
    not
    用于反转逻（不是False变为True.
    11.jpg


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201113170443413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyMzA5Mw==,size_16,color_FFFFFF,t_70#pic_center)


    [32m[2020-10-30 17:55:38,871] [    INFO] - Installing chinese_ocr_db_crnn_mobile module[0m
    [32m[2020-10-30 17:55:38,983] [    INFO] - Module chinese_ocr_db_crnn_mobile already installed in /home/aistudio/.paddlehub/modules/chinese_ocr_db_crnn_mobile[0m
    [32m[2020-10-30 17:55:39,389] [    INFO] - Installing chinese_text_detection_db_mobile module-1.0.3[0m
    [32m[2020-10-30 17:55:39,504] [    INFO] - Module chinese_text_detection_db_mobile-1.0.3 already installed in /home/aistudio/.paddlehub/modules/chinese_text_detection_db_mobile[0m


    人工智能无处不在
    1/飞桨
    搜素引擎：网页、图片、视频、新闻、学术、地图
    信息推荐：新闻、商品、游戏、书籍
    智能助理
    图片识别：人像、用品、动物、交通工具
    用户分析：社交网络、影评、商品评论
    智能图像
    理解
    机器翻译、摘要生成.
    逻辑操作符
    描述
    机器翻译
    and
    网络购物
    如果两个操作数中的任何一个为True则eondition变为Trua
    期
    智能化推荐
    not
    用于反转逻辑（不是False变为True.
    7.jpg



![在这里插入图片描述](https://img-blog.csdnimg.cn/20201113170459188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyMzA5Mw==,size_16,color_FFFFFF,t_70#pic_center)


    [32m[2020-10-30 17:55:45,180] [    INFO] - Installing chinese_ocr_db_crnn_mobile module[0m
    [32m[2020-10-30 17:55:45,181] [    INFO] - Module chinese_ocr_db_crnn_mobile already installed in /home/aistudio/.paddlehub/modules/chinese_ocr_db_crnn_mobile[0m
    [32m[2020-10-30 17:55:45,487] [    INFO] - Installing chinese_text_detection_db_mobile module-1.0.3[0m
    [32m[2020-10-30 17:55:45,489] [    INFO] - Module chinese_text_detection_db_mobile-1.0.3 already installed in /home/aistudio/.paddlehub/modules/chinese_text_detection_db_mobile[0m

三岁的能力有限就这么多啦，有不足的还需要大家多多包涵啦！  

>作者：三岁  
经历：自学python，现在混迹于paddle社区，希望和大家一起从基础走起，一起学习Paddle  
csdn地址：https://blog.csdn.net/weixin_45623093/article/list/3  
我在AI Studio上获得黄金等级，点亮7个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/284366  

>传说中的飞桨社区最差代码人，让我们一起努力！  
记住：三岁出品必是精品 （不要脸系列）
