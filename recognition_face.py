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
