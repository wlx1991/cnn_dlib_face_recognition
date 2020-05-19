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
