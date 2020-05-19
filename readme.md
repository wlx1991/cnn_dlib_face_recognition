## 1前言
在这里我先明确一下开发环境和库版本：

 1. 系统：win10 64位；
 2. IDE：Pycharm2019(免费版)；
 3. Dlib：19.8.1；
 4. Opencv：4.1.1.26；
 5. Keras：2.3.1；
 6. numpy：1.17.4；
 7. scikit-learn：0.19.2；
 8. tensorflow：2.1.0。

&#160; &#160; &#160; &#160;因为在我以前写的一篇关于人脸识别的文章下，有一些同学问了一些问题，还有私信我的。但是，那篇文章说实话，是自己刚开始学习的时候，用“拿来主义”玩儿的，自己当时理解也不到位，所以效果很差。最近想抽一天时间，再下一次，详细阐述一下开发的想法，希望能够帮助做毕设的同学和其他需要的人。
## 2数据
### 2.1数据来源
&#160; &#160; &#160; &#160;一般情况，我们会选择开源的人脸库或者自己网上爬取。首先说说个人的想法，在选择人脸库的时候，我觉得应该注意一下你将来的应用场景，西方人和东方人差距还是挺大的。所以，我还是建议，数据能自己找人拍照最好。
&#160; &#160; &#160; &#160;这里我先写了一个拍照的脚本——gain_face.py。作用是开启摄像头，获取设定好数量的人脸图，每获取一张图像，就通过dlib识别人脸区域，然后将人脸区域以灰度图像保存到指定目录下。

```python
import cv2
import os
import dlib

#创建目录
def CreateFolder(path):
    #去除首位空格
    del_path_space = path.strip()
    #去除尾部'\'
    del_path_tail = del_path_space.rstrip('\\')
    #判读输入路径是否已存在
    isexists = os.path.exists(del_path_tail)
    if not isexists:
        os.makedirs(del_path_tail)
        return True
    else:
        return False

#提取并保存人脸
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):

    #检查输入路径是否存在——不存在就创建
    CreateFolder(path_name)
    cv2.namedWindow(window_name)
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    #导入正脸探测器（实例化）
    detector = dlib.get_frontal_face_detector()

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    #人脸个数
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  #读取一帧数据
        #图像获取失败，退出
        if not ok:
            break
        #将当前桢图像转换成灰度图像
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #使用探测器识别图像中的人脸，形成一个人脸列表
        face_dets = detector(img_gray, 1)
        det = None
        #如果没有人脸就退出本次循环
        if len(face_dets) == 0:
            continue
        elif len(face_dets) > 1:
            #只要最大的那个人脸
            temp_area = 0
            temp = 0
            for i, face_area in enumerate(face_dets):
                if (face_area.right() - face_area.left()) * (face_area.bottom() - face_area.top()) > temp_area:
                    temp_area = (face_area.right() - face_area.left()) * (face_area.bottom() - face_area.top())
                    temp = i
            det = face_dets[temp]
        else:
            det = face_dets[0]
        # 提取人脸区域
        face_top = det.top() if det.top() > 0 else 0
        face_bottom = det.bottom() if det.bottom() > 0 else 0
        face_left = det.left() if det.left() > 0 else 0
        face_right = det.right() if det.right() > 0 else 0
        # 将当前帧保存为图片
        img_name = '%s\%d.jpg' % (path_name, num)

        #face_img = frame[face_top:face_bottom, face_left:face_right]
        face_img = img_gray[face_top:face_bottom, face_left:face_right]          #保存灰度人脸图
        cv2.imwrite(img_name, face_img)
        #人脸数量加一
        num += 1
        #画出矩形框的时候稍微比识别的脸大一圈
        cv2.rectangle(frame, (det.left() - 10,det.top() - 10), (det.right() + 10, det.bottom() + 10), color, 2)
        #显示当前捕捉到了多少人脸图片
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'num:%d' % (num), (det.left() + 30, det.top() + 30), font, 1, (255, 0, 255), 4)
        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): break
        # 显示图像
        cv2.imshow(window_name, frame)
        #按键盘‘Q’中断采集
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    while True:
        print("是否录入员工信息(Yes or No)?")
        if input() == 'Yes':
            #员工姓名(要输入英文，汉字容易报错)
            new_user_name = input("请输入您的姓名：")

            print("请看摄像头！")

            #采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢
            window_name = 'xinxicaiji'           #图像窗口
            camera_id = 0                        #相机的ID号
            images_num = 50                      #采集图片数量
            # 图像保存位置
            path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face' + '/' + new_user_name

            CatchPICFromVideo(window_name, camera_id, images_num, path)
        else:
            break
```
&#160; &#160; &#160; &#160;这段代码没啥可说的，把保存路径设置好，把需要的数据量设置好。
### 2.2数据处理
#### 2.2.1第一步
&#160; &#160; &#160; &#160;数据处理是数据分析的前提，真的是关键之一。一般情况下包括：数据清洗，数据标准化等。在这里就是清洗一下人脸图中角度很偏的、亮度差的图像；然后，为数据进行标记；最后将数据分为训练集、验证集和测试集。
&#160; &#160; &#160; &#160;本次试验，我获取了三个人的人脸图像，每个人有51张图像。我将每一类数据中的50张设定为训练集，1张设定为测试集。我又从50张训练集中拿出15张作为验证集，符合（7:3）。我将训练集重新建立了测试目录。如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200517123729312.png)
&#160; &#160; &#160; &#160;所以，代码中我就写了两个相似的特征提取的函数，一个是训练集的，一个是测试集的。最后，我把训练集和测试集数据都以npz格式保存了。
&#160; &#160; &#160; &#160;这个脚本的作用就是从训练目录和测试目录下提取数据。因为每一类数据都有一个以名字命名的目录，所以每一张图片的标签就是自己的目录名，然后将目录标签转换为对应的数字标签。
&#160; &#160; &#160; &#160;最后，训练数据和标签保存在video_face_train.npz文件中，同时我将每一类的目录名组合成列表也保存在该文件中。`np.savez(npz_path, face_imgs=images_train, face_labels=labels_train,face_names = class_list)`。测试数据和标签保存在video_face_test.npz文件中。`np.savez(npz_path, face_imgs=images_test, face_labels=labels_test)`。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200517125843593.png)
下面就是video_dataset.py的脚本代码。

```python
import numpy as np
import dlib
import cv2
import os

#############################
#每一类50张训练图像，
#############################

#获得当前项目的根目录——也就是当前脚本的目录的上一级目录
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'
#导入正脸探测器（实例化）
detector = dlib.get_frontal_face_detector()

#图像大小64*64
img_size = 64

#将图像压缩为64*64
def reszie_image(image,height=img_size,width=img_size):
    top,bottom,left,right = 0,0,0,0
    ##############如果图像不是正方形################
    #获取图像大小
    h,w = image.shape
    #对于长宽不一的，取最大边长
    longest_edge = max(h,w)
    #计算较短的边需要增加的像素值
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #############################################
    if top==0 and bottom==0 and left==0 and right==0:
        return cv2.resize(image,(height,width))
    else:
        #定义边界填充颜色
        BLACK = [0, 0, 0]
        #为图像增加边界
        constant_img = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
        return cv2.resize(constant_img,(height,width))

#为每一类数据赋予唯一的标签值
def label_id(label,class_list,class_num):
    for i in range(class_num):
        if label == class_list[i]:
            return i

#获取训练照片的人脸数据
def TrainFeature(root_train_path,npz_path):   #root_train_path:训练数据的根目录
    #训练数据集
    images_train = []
    #训练数据的标签集
    labels_train = []

    #类别数量
    class_list = os.listdir(root_train_path)
    class_num = len(class_list)

    for img_dir in class_list:
        for img_name in os.listdir(root_train_path+'\\'+img_dir):
            image = cv2.imread(root_train_path+'\\'+img_dir+'\\'+img_name,cv2.IMREAD_GRAYSCALE)
            image = reszie_image(image)

            images_train.append(image)
            labels_train.append(img_dir)
    images_train = np.array(images_train)
    labels_train = np.array([label_id(label,class_list,class_num) for label in labels_train])

    #读取已有数据库，如果没有就创建数据库
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path, face_imgs=images_train, face_labels=labels_train,face_names = class_list)
    else:
        # 保存数据
        np.savez(npz_path, face_imgs=images_train, face_labels=labels_train,face_names = class_list)
        npz_dates.close()

#获取测试数据
def TestFeature(root_test_path,npz_path):   #root_test_path:测试数据的根目录
    #训练数据集
    images_test = []
    #训练数据的标签集
    labels_test = []
    #数据的人名


    #类别数量
    class_list = os.listdir(root_test_path)
    class_num = len(class_list)

    for img_dir in class_list:
        for img_name in os.listdir(root_test_path+'\\'+img_dir):
            image = cv2.imread(root_test_path+'\\'+img_dir+'\\'+img_name,cv2.IMREAD_GRAYSCALE)
            image = reszie_image(image)

            images_test.append(image)
            labels_test.append(img_dir)
    images_test = np.array(images_test)
    labels_test = np.array([label_id(label,class_list,class_num) for label in labels_test])

    # 读取已有数据库，如果没有就创建数据库
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path, face_imgs=images_test, face_labels=labels_test)
    else:
        #保存数据
        np.savez(npz_path, face_imgs=images_test, face_labels=labels_test)
        npz_dates.close()

if __name__ == '__main__':
    print('#################测试###################')
    #训练数据的根目录
    root_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face_train'
    npz_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_train.npz'
    #测试数据的根目录
    root_test_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face_test'
    npz_test_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_test.npz'
    TrainFeature(root_train_path,npz_train_path)
    TestFeature(root_test_path,npz_test_path)

    data_train = np.load(npz_train_path)
    print(data_train['face_imgs'].shape)
    print(data_train['face_labels'].shape)
    print(data_train['face_names'])
    data_test = np.load(npz_test_path)
    print(data_test['face_imgs'].shape)
    print(data_test['face_labels'])

```
&#160; &#160; &#160; &#160;对了，这里在说一点，因为目录中保存的图像是灰度图像，所以使用opencv获取图像时，在imread第二个参数选择`cv2.IMREAD_GRAYSCALE`，否则你提取的图像还是3通道的。示例：`image = cv2.imread(‘*图片路径*’,cv2.IMREAD_GRAYSCALE)`
#### 2.2.2第二步
&#160; &#160; &#160; &#160;在第一步中，我们将训练数据和测试数据都保存在了两个npz文件中。这一步就是将数据提取出来，然后将训练数据分成训练集和验证集，接着把数据的标签进行One-hot编码，最后把特征数据归一化处理。（这些都是为了模型训练做准备，可能有人要问，我上一步已经把数据弄好了，直接划分训练和验证，然后one-hot编码等不就行了，为何要保存到文件里面？就是为了之后的训练方便，不用每一次训练前都要重新从目录中提取特征。）
&#160; &#160; &#160; &#160;load_datas.py代码如下。因为数据是单通道，所以加载数据时，重排数据维度时`img_channels=1`。

```python
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

```
## 3模型训练
&#160; &#160; &#160; &#160;在该脚本中会建立一个8层的神经网络，因为之前的文章就是用这个模型，所以也就没改。但是，我认为不需要怎么多层，有个三是层就够了，有兴趣的同学可以自己修改一下。
&#160; &#160; &#160; &#160;我们使用的CNN（卷积神经网络），在模型建立函数`def build_model(self, dataset,nb_classes=3)`中，我简单的说明了每一层网络的作用。在`def train（）`函数中，有两种训练方式，一种是直接用我们现在的数据进行训练，不进行数据的扩展；另一种就是每一次迭代都会随机扩展出`batch_size`数量的新数据，具体说明可以查看`ImageDataGenerator()`的用法。
&#160; &#160; &#160; &#160;cnn_model.py代码如下。

```python
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from losshistory import LossHistory

#获得当前项目的根目录——也就是当前脚本的目录的上一级目录
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'

#CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
        self.history = LossHistory()

        # 建立模型
    def build_model(self, dataset,nb_classes=3):

        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  #第1层:2维卷积层
        self.model.add(Activation('relu'))                              #第1层:激活函数

        self.model.add(Convolution2D(32, 3, 3))                         #第2层:2维卷积层
        self.model.add(Activation('relu'))                              #第2层:激活函数
        #池化层的作用:
        #1.invariance(不变性):translation(平移),rotation(旋转),scale(尺度)
        #2.保留主要特征的同时，进行了降维处理，防止过拟合，提高模型的泛化能力
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #第3层:池化层
        self.model.add(Dropout(0.25))                                   #第3层:Dropout——这一层中的每个节点有25%的概率失活

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))     #第4层:2维卷积层
        self.model.add(Activation('relu'))                              #第4层:激活函数

        self.model.add(Convolution2D(64, 3, 3))                         #第5层:2维卷积层
        self.model.add(Activation('relu'))                              #第5层:激活函数

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  #第6层:池化层
        self.model.add(Dropout(0.25))                                   #第6层:Dropout

        self.model.add(Flatten())                                       #第7层:Flatten()——多维输入一维化
        self.model.add(Dense(512))                                      #第7层:Dense层,又被称作全连接层
        self.model.add(Activation('relu'))                              #第7层:激活函数
        self.model.add(Dropout(0.5))                                    #第7层:Dropout

        self.model.add(Dense(nb_classes))                               #第8层:Dense层
        self.model.add(Activation('softmax'))                           #第8层:分类层，输出最终结果

        #输出模型概况
        self.model.summary()

    # 训练模型
    def train(self, train_images,train_labels,valid_images,valid_labels, batch_size=20, nb_epoch=10, data_augmentation=False):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量。
        if not data_augmentation:
            self.model.fit(train_images,
                           train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(valid_images,valid_labels),
                           shuffle=True,
                           callbacks=[self.history])
        #使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,               # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,                # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,    # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,     # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,                    # 是否对输入数据施以ZCA白化
                rotation_range=20,                      # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,                  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,                 # 同上，只不过这里是垂直
                horizontal_flip=True,                   # 是否进行随机水平翻转
                vertical_flip=False)                    # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(train_images, train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(valid_images, valid_labels),
                                     callbacks=[self.history])

    MODEL_PATH = os_path + '/data/model/aggregate.face.model.h5'
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
    def evaluate(self, test_images,test_labels):
        score = self.model.evaluate(test_images, test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

if __name__ == '__main__':
    from load_datas import Dataset

    #训练数据路径
    train_npz_path = os_path + r'\data\npz\video_face_train.npz'
    #测试数据路径
    test_npz_path = os_path + r'\data\npz\video_face_test.npz'
    #训练图像保存的路径
    train_img_dir = os_path + r'\data\video_face_train'
    #实例化数据对象
    datasets = Dataset(train_npz_path, test_npz_path, train_img_dir)
    #导入训练和验证数据
    train_images,valid_images,train_labels,valid_labels = datasets.load_train_valid()
    #导入测试数据
    test_images, test_labels = datasets.load_test()

    #cnn模型对象实例化
    cnn_model = Model()
    #创建模型
    cnn_model.build_model(datasets,nb_classes=datasets.user_num)        #datasets.user_num种类数量
    #训练模型
    cnn_model.train(train_images,train_labels,valid_images,valid_labels,batch_size=20,nb_epoch=10)
    #评估模型
    cnn_model.evaluate(test_images,test_labels)
    #绘制损失曲线和准确率曲线
    cnn_model.history.loss_plot('epoch')
    #保存模型
    cnn_model.save_model()

    # #加载模型
    # model_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\model\aggregate.face.model.h5'
    # cnn_model = load_model(model_path)
    # score = cnn_model.model.evaluate(test_images, test_labels, verbose=1)
    # print("%s: %.2f%%" % (cnn_model.model.metrics_names[1], score[1] * 100))
```
&#160; &#160; &#160; &#160;在上面代码中增加了一个绘制损失函数和准确度的方法。需要再另外添加一个脚本losshistory.py。

```python
import matplotlib.pyplot as plt
import keras
################################
#收集训练时候的accuracy和loss
################################

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
```
&#160; &#160; &#160; &#160;如果大家想让训练和验证的损失、准确度曲线再两张图片上显示，可以修改losshistory.py中的`def loss_plot(self, loss_type)`函数，很简单。
&#160; &#160; &#160; &#160;好训练完成，同时，我们也把模型保存到了model目录下。
## 4模型使用
&#160; &#160; &#160; &#160;接下来就是使用我们训练的模型，去识别新的图像，这里想强调一点，就是新的图像要和你训练时的图像保持一致：包括尺寸、格式都要一致。
&#160; &#160; &#160; &#160;下面贴出最后的识别脚本——recognition.py。在获取训练图像时，我们只想要图像中最前面的那个人脸，但在识别的时候，图像中的所有人脸都应该被识别。

```python
from keras.models import load_model
from keras import backend as K
from video_datasets import reszie_image,img_size
import cv2

def face_predict(image,model,class_list,classes):
    ##############提取人脸特征###################
    # 灰度处理
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #使用探测器识别图像中的人脸，形成一个人脸列表
    face_dets = detector(img_gray, 1)
    #如果没有人脸就将图像原路返回
    if len(face_dets) == 0:
        return image
    else:
        for det in face_dets:
            #提取人脸区域
            face_top = det.top() if det.top() > 0 else 0
            face_bottom = det.bottom() if det.bottom() > 0 else 0
            face_left = det.left() if det.left() > 0 else 0
            face_right = det.right() if det.right() > 0 else 0

            face_temp = img_gray[face_top:face_bottom, face_left:face_right]         #灰度图
            face_img = None
            #压缩图像为64*64
            little_face = reszie_image(face_temp)
            ####################################################
            # 依然是根据后端系统确定维度顺序
            if K.image_data_format() == 'channels_first':
                face_img = little_face.reshape((1,1,img_size, img_size))               #与模型训练不同，这次只是针对1张图片进行预测
            elif K.image_data_format() == 'channels_last':
                face_img = little_face.reshape((1, img_size, img_size, 1))
            #浮点并归一化
            face_img = face_img.astype('float32')
            face_img /= 255

            #给出输入属于各个类别的概率
            result_probability = model.predict_proba(face_img)
            #print('result:', result_probability)

            # 给出类别预测(改）
            if max(result_probability[0]) >= 0.9:
                result = model.predict_classes(face_img)
                #print('result:', result)
                #返回类别预测结果
                faceID = result[0]
            else:
                faceID = -1
            #画框
            cv2.rectangle(image, (face_left - 10, face_top - 10), (face_right + 10, face_bottom + 10), color,
                          thickness=2)
            #face_id判断
            if faceID in classes:
                # 文字提示是谁
                cv2.putText(image, class_list[faceID],
                            (face_left, face_top - 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
            else:
                # 文字提示是谁
                cv2.putText(image, 'None ',
                            (face_left, face_top - 30),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
    return image

if __name__ == '__main__':
    import dlib
    import numpy as np
    #类别列表
    npz_train_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\npz\video_face_train.npz'
    data_train = np.load(npz_train_path)
    class_list = data_train['face_names']
    classes = [i for i in range(len(class_list))]
    #加载模型
    model_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\model\aggregate.face.model.h5'
    cnn_model = load_model(model_path)
    #框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    # 导入正脸探测器（实例化）
    detector = dlib.get_frontal_face_detector()
    # 循环检测识别人脸
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if ret is False:
            continue
        else:
            frame = face_predict(frame,cnn_model,class_list,classes)
        cv2.imshow("login", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

```
## 5试验结果
&#160; &#160; &#160; &#160;在最后把试验结果贴一下。

```python
Epoch 9/10

 20/105 [====>.........................] - ETA: 0s - loss: 1.4413e-04 - accuracy: 1.0000
 40/105 [==========>...................] - ETA: 0s - loss: 2.9561e-04 - accuracy: 1.0000
 60/105 [================>.............] - ETA: 0s - loss: 2.8445e-04 - accuracy: 1.0000
 80/105 [=====================>........] - ETA: 0s - loss: 0.0019 - accuracy: 1.0000    
100/105 [===========================>..] - ETA: 0s - loss: 0.0015 - accuracy: 1.0000
105/105 [==============================] - 1s 13ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0295 - val_accuracy: 0.9778
Epoch 10/10

 20/105 [====>.........................] - ETA: 0s - loss: 3.3582e-04 - accuracy: 1.0000
 40/105 [==========>...................] - ETA: 0s - loss: 4.0205e-04 - accuracy: 1.0000
 60/105 [================>.............] - ETA: 0s - loss: 0.0070 - accuracy: 1.0000    
 80/105 [=====================>........] - ETA: 0s - loss: 0.0054 - accuracy: 1.0000
100/105 [===========================>..] - ETA: 0s - loss: 0.0044 - accuracy: 1.0000
105/105 [==============================] - 1s 13ms/step - loss: 0.0042 - accuracy: 1.0000 - val_loss: 0.0402 - val_accuracy: 0.9778
#这是测试的结果
3/3 [==============================] - 0s 6ms/step
accuracy: 100.00%
```
&#160; &#160; &#160; &#160;这是训练过程中的训练集和验证集的损失度和准确度曲线。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200517153238913.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaWxpeGluODg=,size_16,color_FFFFFF,t_70)
##  6总结
&#160; &#160; &#160; &#160;本次试验共有三类人脸数据，每类数据51张，其中50张中35张是训练集，15张是验证集；1张是测试集。所以，大家要明确验证集和测试集是不同的数据集，不要把验证集哪来再次到测试集使用，那是没有意义的。另外，从测试结果上看，效果好像挺好。但是，在实验过程中，发现应用效果还是不好，在这里稍微分析一下，一是我的数据集太少，太少就导致模型不能充分学习到人脸的角度，亮度等因素特征；二我的数据集中有两类数据是去年拍的，所以现在训练完了后，识别现在的人脸效果就差，所以实际应用中是不是应该把每次识别到的人脸再次保存到数据库中，作为训练数据；三如果使用二的方法，那么人脸识别就不能作为唯一的识别手段，比如：我们乘坐火车过安检的时候，你先刷身份证，然后在人脸认证。目的就在于用身份证识别保证不把你识别错误，因为身份证识别可以说是百分之百正确，然后再刷脸进行二次验证，这个时候得到的人脸图像可以放心的存入对应的人脸数据库中，为之后的模型更新补充了新的数据。
&#160; &#160; &#160; &#160;以上就是本人的拙见，希望能帮助需要的人。我把代码上传到[https://download.csdn.net/download/weilixin88/12431198](https://download.csdn.net/download/weilixin88/12431198)，有需要的可以下载一下。
