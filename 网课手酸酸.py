#!/usr/bin/env python
# coding: utf-8

# ### 网课看完笔油见底，两手酸酸，头皮麻麻，本子满满密密麻麻。
# ### 你还在为网课抄不完的笔记烦恼嘛？
# ### 你还在为网课那弄不完的无底洞头疼嘛?
# ## 你们的救星来了！！！！

# ### 这里是大家的三岁，万能小白三岁，大家的苦恼网课笔记手麻麻，用paddleOCR解决啦！！！

# ### 让我们一起来看看叭！
# #### 有关的环境配置
# 该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper

# In[3]:


## 配置有关的环境
get_ipython().system('hub install chinese_ocr_db_crnn_mobile==1.1.0')
get_ipython().system('pip install shapely')
get_ipython().system('pip install pyclipper')


# 本次使用的OCR模型是haddlehubOCR chinese_ocr_db_crnn_server的1.1.0版本。
# ## 模型概述
# chinese_ocr_db_crnn_server Module用于识别图片当中的汉字。其基于chinese_text_detection_db_server检测得到的文本框，继续识别文本框中的中文文字。之后对检测文本框进行角度分类。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个通用的OCR模型，支持直接预测。
# ![](https://ai-studio-static-online.cdn.bcebos.com/d75322a86d1042ae93a45c44e0b51e755038be360c9e4932914f7d271f47c280)
# 

# ### 文件处理

# In[4]:


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


# ### 视频分割

# In[1]:


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


# In[ ]:


# !hub run chinese_ocr_db_crnn_mobile --input_path "1596726259(1).jpg" # 使用Linux环境使用OCR


# In[2]:


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


# 目前三岁的能力就这么多啦，有不足的还需要大家多多包涵啦！

# #  作者简介
# ## 作者：三岁
# ### 大数据助理工程师
# ### 深度学习、机器学习在学、爱好者
# ### CSDN地址：https://blog.csdn.net/weixin_45623093
# 希望大家多多关注！
