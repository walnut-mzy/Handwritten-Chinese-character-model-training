#C:\Users\mzy\Desktop\data_train\train

import glob
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import setting
import cv2
#import numpy as np
#一个小问题
#AttributeError: 'Tensor' object has no attribute 'numpy'
#
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image=image.numpy()
    image=zaodian(image)
    #将维度转为三维并且翻转至RGB图片
    image=[tf.squeeze(image,axis=-1) for _ in range(3)]
    image=tf.transpose(image,[1,2,0])
    # # #图片展示
    # image = image.numpy()
    # cv2.imshow("aa",image)
    # cv2.waitKey(0)
    return image
def get_train(pic):
    img=load(pic)
    if tf.random.uniform(()) > 0.5:  # 从均匀分布中返回随机值 如果大于0.5就执行下面的随机翻转
        img = tf.image.flip_left_right(img)
    # img = tf.cast(img, tf.float32) / 127.5 - 1
    return img
def zaodian(binImg):
    a=0
    while a<=20:
        a+=1
        pixdata = binImg
        width, height ,_= binImg.shape
        for y in range(1, height- 1):
            for x in range(1, width- 1):
                count = 0
                sudi=0
                if pixdata[x, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x, y + 1] == sudi:
                    count = count + 1
                if pixdata[x - 1, y] == sudi:
                    count = count + 1
                if pixdata[x + 1, y] == sudi:
                    count = count + 1
                if pixdata[x - 1, y - 1] ==sudi:
                    count = count + 1
                if pixdata[x - 1, y + 1] == sudi:
                    count = count + 1
                if pixdata[x + 1, y - 1]== sudi:
                    count = count + 1
                if pixdata[x + 1, y + 1] == sudi:
                    count = count + 1
                if count >4:
                    pixdata[x, y] =0
            for i in range(1, height):

                pixdata[1, i] = 0
                pixdata[width-1,i]=0
            for j in range(1, width):

                pixdata[j, 1] = 0
                pixdata[j,height-1]=0
    return pixdata
def train():
    list_label=[]
    list_val=[]
    file=glob.glob(setting.train_path)
    for i in file:
        for j in glob.glob(i+"/*.png"):
            list_label.append(int(i[-4:]))
            list_val.append(get_train(j))
    list_label=tf.one_hot(list_label,depth=3754)
    dataset = tf.data.Dataset.from_tensor_slices((list_val,list_label))
    dataset = dataset.shuffle(setting.BUFFER_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE).batch(setting.BATCH_SIZE)
    print(dataset)
    return dataset
def test():
    list_label = []
    list_val = []
    file = glob.glob(setting.test_path)
    for i in file:
        for j in glob.glob(i + "/*.png"):
            list_label.append(int(i[-4:]))
            list_val.append(get_train(j))
    list_label = tf.one_hot(list_label, depth=3754)
    dataset = tf.data.Dataset.from_tensor_slices((list_val, list_label))
    dataset = dataset.shuffle(setting.BUFFER_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE).batch(setting.BATCH_SIZE)
    print(dataset)
    return dataset
#load("C:/Users/mzy/Desktop/data_train/train/03742/20.png")
test()