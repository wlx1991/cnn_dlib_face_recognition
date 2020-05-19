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



