################
#通过摄像头拍照获取人的脸部信息
################

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
            path = r'C:\Users\wlx\Documents\py_study\deeplearning\cnn_face_recogntion\data\video_face_train' + '/' + new_user_name

            CatchPICFromVideo(window_name, camera_id, images_num, path)
        else:
            break
