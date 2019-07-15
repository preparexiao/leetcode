import tensorflow as tf
import numpy as np
import os
import pandas as pd

#https://blog.csdn.net/lingtianyulong/article/details/80495084
#panda用法： https://blog.csdn.net/xz1308579340/article/details/81106310
#kaggle数据集： https://www.kaggle.com/c/aptos2019-blindness-detection/data

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    """

    :param image: 图片路径
    :param label:图片标签
    :param resize_w: 图片宽
    :param resize_h: 图片长
    :param batch_size: batch大小设置
    :param capacity: capacity大小
    :return: return 图片batch以及label
    """
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])#实现一个输入队列
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)

    # resize
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # (x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)#对图片进行标准化
    print("stand image",image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch

def get_files2(filename,N):
    """

    :param filename: train,test文件夹
    :param N: 取前N条训练数据
    :return: 返回相应的训练数据路径以及标签
    """
    class_train = []
    label_train = []

    print(filename+'/'+"train.csv")
    lable_data=pd.read_csv(filename+"train.csv")
    lable_data_head=lable_data.head(N)
    # print(lable_data_head)
    id_code=list(lable_data_head['id_code'])
    diagnosis = list(lable_data_head['diagnosis'])
    # print("list id_code",id_code)
    # print("list diagnosis",diagnosis)

    # for train_class in os.listdir(filename):
    for pic in os.listdir(filename+"train_data"):
        # print("os.listdir(filename):",os.listdir(filename))
        class_train.append(filename + '/' + pic)
        pic=pic.split('.')[0]

        if pic in id_code:
            kind = id_code.index(pic)
            kind=diagnosis[kind]
        else:
            kind=-1
        label_train.append(kind)
    kind = kind + 1
    temp = np.array([class_train, label_train])
    temp = temp.transpose()  # transpose()代表矩阵转置
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # print(label_list)
    return image_list, label_list

# path = r'C:/Users/xiaomin/Desktop/pre/leetcode/tensorflow-pre/test_image/'
# image_list,lable_list=get_files(path)
path = r'C:/Users/xiaomin/Desktop/pre/leetcode/tensorflow-pre/test_image2/'
image_list,lable_list=get_files2(path,10)
image_batches,label_batches = get_batches(image_list,lable_list,32,32,16,20)
print("ok")
print(image_batches)
print(label_batches)
#定义输入层
tf_X=tf.placeholder(tf.float32,[16,32,32,3])
tf_Y = tf.placeholder(tf.int64,[16,])

# 卷积层+激活层
conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 1, 10]))
conv_filter_b1 =  tf.Variable(tf.random_normal([10]))
relu_feature_maps1 = tf.nn.relu(\
                tf.nn.conv2d(tf_X, conv_filter_w1,strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1)

# 池化层
max_pool1 = tf.nn.max_pool(relu_feature_maps1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
print(max_pool1)

# 卷积层
conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 10, 5]))
conv_filter_b2 =  tf.Variable(tf.random_normal([5]))
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2
print(conv_out2)
# 卷积层
conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 10, 5]))
conv_filter_b2 =  tf.Variable(tf.random_normal([5]))
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2