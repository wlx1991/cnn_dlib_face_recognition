import numpy as np
import os
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#获得当前项目的根目录——也就是当前脚本的目录的上一级目录
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'
IMAGE_SIZE = 64

class Dataset:
    def __init__(self,train_npz_path,test_npz_path,train_img_dir):

        #训练数据集加载路径
        self.train_npz_path = train_npz_path
        #测试数据集加载路径
        self.test_npz_path = test_npz_path
        #图像种类
        self.user_num = len(os.listdir(train_img_dir))
        #当前库采用的维度顺序
        self.input_shape = None

    #加载训练数据集——并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load_train_valid(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1):
        #数据种类
        nb_classes = self.user_num
        #加载数据集到内存
        datas = np.load(self.train_npz_path)
        images,labels = datas['face_imgs'],datas['face_labels']
        #变换labels的维度
        labels = labels.reshape(len(labels),1)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images,labels, test_size=0.3,random_state=1)
        # 当前的维度顺序如果为'th'=='channels_first'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        #输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)

        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        valid_images /= 255

        return train_images,valid_images,train_labels,valid_labels

    #加载测试数据集
    def load_test(self,img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1):
        # 数据种类
        nb_classes = self.user_num
        #加载测试数据集到内存
        datas = np.load(self.test_npz_path)
        test_images, test_labels = datas['face_imgs'], datas['face_labels']
        #变换test_labels的维度
        test_labels = test_labels.reshape(len(test_labels),1)
        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == 'channels_first':
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
        else:
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)

        #输出训练集、验证集、测试集的数量
        print(test_images.shape[0], 'test samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        #像素数据浮点化以便归一化
        test_images = test_images.astype('float32')
        #将其归一化,图像的各像素值归一化到0~1区间
        test_images /= 255

        return test_images,test_labels


if __name__ == '__main__':
    train_npz_path = os_path + r'\data\npz\video_face_train.npz'
    test_npz_path = os_path + r'\data\npz\video_face_test.npz'
    train_img_dir = os_path + r'\data\video_face_train'
    #试一试
    datasets = Dataset(train_npz_path,test_npz_path,train_img_dir)
    train_images,valid_images,train_labels,valid_labels = datasets.load_train_valid()
    test_images, test_labels = datasets.load_test()
    print(train_labels)
    print(test_labels)
