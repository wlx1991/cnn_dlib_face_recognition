import numpy as np
import dlib
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator,img_to_array


#############################
#每一类实际只有一张图像，然后使用ImageDataGenerator()虚假扩展训练数据
#############################


#获得当前项目的根目录——也就是当前脚本的目录的上一级目录
os_path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion'
#导入正脸探测器（实例化）
detector = dlib.get_frontal_face_detector()
# #导入人脸关键点识别器
# predictor = dlib.shape_predictor(os_path + '/data/model/shape_predictor_68_face_landmarks.dat')
# #导入人脸检测模型
# model = dlib.face_recognition_model_v1(os_path + '/data/model/dlib_face_recognition_resnet_model_v1.dat')
#图像大小64*64
img_size = 64
#扩展的样本数
num_samples = 100

#将图像压缩为64*64
def reszie_image(image,height=img_size,width=img_size):
    top,bottom,left,right = 0,0,0,0
    ##############如果图像不是正方形################
    #获取图像大小
    h,w,_ = image.shape
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

#获取训练照片的人脸数据
def TrainFeature(img_dir,img_name,npz_path):   #img_dir:图像存放文件夹的路径;img_name:图像的名称;npz_path:特征与标签保存路径
    #获取类别数量
    users = os.listdir(img_dir)
    num_classes = len(users)
    #最新的类别数量就是新加入人脸的标签
    face_label = num_classes
    ##############提取人脸特征###################
    img = cv2.imread(img_dir+'/'+img_name)      #opencv度到的图像是BGR
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #灰度处理
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #使用探测器识别图像中的人脸，形成一个人脸列表
    face_dets = detector(img_gray, 1)
    det = None
    if len(face_dets) == 0:
        return False
    elif len(face_dets) > 1:
        # 只要最大的那个人脸
        temp_area = 0
        temp = 0
        for i,face_area in enumerate(face_dets):
            if (face_area.right()-face_area.left())*(face_area.bottom()-face_area.top()) > temp_area:
                temp_area = (face_area.right()-face_area.left())*(face_area.bottom()-face_area.top())
                temp = i
        det = face_dets[temp]
    else:
        det = face_dets[0]

    #提取人脸区域
    face_top = det.top() if det.top()>0 else 0
    face_bottom = det.bottom() if det.bottom()>0 else 0
    face_left = det.left() if det.left()>0 else 0
    face_right = det.right() if det.right()>0 else 0

    face_img = img[face_top:face_bottom,face_left:face_right]
    #改变face_img的维度
    face_img = img_to_array(face_img).reshape(1,face_img.shape[0],face_img.shape[1],3)

    #增加样本
    datagen = ImageDataGenerator(
        featurewise_center=False,           #是否使输入数据去中心化
        samplewise_center=False,            #是否使输入数据的每个样本均值为0
        featurewise_std_normalization=False,#是否数据标准化
        samplewise_std_normalization=False, #是否将每个样本除以自身的标准差
        zca_whitening=False,                #是否对输入数据进行ZCA白化
        rotation_range=20,                  #图像随机转动角度（范围为0~180）
        width_shift_range=0.2,              #图像水平偏移的幅度（图像宽度的占比：0~1）
        height_shift_range=0.2,             #图像垂直偏移的幅度
        horizontal_flip=True,               #进行随机水平翻转
        vertical_flip=False                 #进行垂直翻转
    )
    #激活datagen
    datagen.fit(face_img)
    #增加样本
    data_iter = datagen.flow(face_img,batch_size=1)
    #计数标志
    count = 0
    #存放新样本特征的列表
    face_img_list = []
    #存放新样本标签的列表
    face_img_labels = []
    for x_batch in data_iter:
        count += 1
        #压缩图像为64*64
        little_face = reszie_image(x_batch[0])
        face_img_list.append(little_face)
        face_img_labels.append(face_label)
        #增加样本的数量
        if count > num_samples-1:
            break
    #将列表转换为numpy.array()类型
    face_img_list = np.array(face_img_list)
    face_img_labels = np.array(face_img_labels).reshape(1,len(face_img_labels))     #并调整维度
    #读取已有数据库，如果没有就创建数据库
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path,face_imgs = face_img_list,face_labels = face_img_labels)
    else:
        #先把人脸的特征数据和标签数据读出
        npz_imgs = npz_dates['face_imgs']
        npz_labels = npz_dates['face_labels']
        #把新的特征数据和标签数据扩展到以前数据之后
        new_npz_imgs = np.vstack((npz_imgs, face_img_list))
        new_npz_labels = np.hstack((npz_labels,face_img_labels))
        #保存数据
        np.savez(npz_path,face_imgs = new_npz_imgs,face_labels = new_npz_labels)
        npz_dates.close()
    return True

#生成用于测试的人脸数据
def TestFeature(img_dir,img_name,npz_path):     #img_dir:图像存放文件夹的路径;img_name:图像的名称;npz_path:特征与标签保存路径
    # 获取类别数量
    users = os.listdir(img_dir)
    num_classes = len(users)
    #最新的类别数量就是新加入人脸的标签
    face_label = num_classes
    ##############提取人脸特征###################
    img = cv2.imread(img_dir + '/' + img_name)  # opencv度到的图像是BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用探测器识别图像中的人脸，形成一个人脸列表
    face_dets = detector(img_gray, 1)
    det = None
    if len(face_dets) == 0:
        return False
    elif len(face_dets) > 1:
        # 只要最大的那个人脸
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

    face_img = img[face_top:face_bottom, face_left:face_right]
    #改变face_img的数据类型
    face_img = img_to_array(face_img)
    #压缩图像为64*64
    little_face = reszie_image(face_img)

    face_img_list = []
    face_img_labels = []
    face_img_list.append(little_face)
    face_img_labels.append(face_label)
    # 将列表转换为numpy.array()类型
    face_img_list = np.array(face_img_list)
    face_img_labels = np.array(face_img_labels).reshape(1,len(face_img_labels))     #并调整维度
    #读取已有数据库，如果没有就创建数据库
    try:
        npz_dates = np.load(npz_path)
    except:
        np.savez(npz_path, face_imgs=face_img_list, face_labels=face_img_labels)
    else:
        # 先把人脸的特征数据和标签数据读出
        npz_imgs = npz_dates['face_imgs']
        npz_labels = npz_dates['face_labels']
        # 把新的特征数据和标签数据扩展到以前数据之后
        new_npz_imgs = np.vstack((npz_imgs, face_img_list))
        new_npz_labels = np.hstack((npz_labels, face_img_labels))
        # 保存数据
        np.savez(npz_path, face_imgs=new_npz_imgs, face_labels=new_npz_labels)
        npz_dates.close()
    return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # img_dir = os_path + r'\data\train_cnn'
    # img_name = 'Helen_Clark_0003.jpg'
    # npz_path = os_path + r'\data\npz\features_labels.npz'
    #
    # flag = TrainFeature(img_dir,img_name,npz_path)
    # print(flag)

    img_dir = os_path + r'\data\test_cnn'
    img_name = 'Helen_Clark_0004.jpg'
    npz_path = os_path + r'\data\npz\test_features_labels.npz'

    flag = TestFeature(img_dir, img_name, npz_path)
    print(flag)

    data = np.load(npz_path)
    print(data['face_imgs'].shape)
    print(data['face_labels'].shape)


