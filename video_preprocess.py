import cv2
import os
im_dir = 'D:\\data_test\\CASME2_RAW_selected\\'
emotion_list = ['happiness', 'disgust', 'fear', 'sadness', 'anger']

def video_crop(file_dir):
    for dic in os.listdir(file_dir):
        path = os.path.join(file_dir, dic)
        for dictonary in os.listdir(path):
            seq_order = []
            all_path_order = []
            jpg_path = os.path.join(path, dictonary)
            print(jpg_path)
            for jpg_name in os.listdir(jpg_path):
                seq = int(jpg_name[3:].split('.')[0])
                seq_order.append(seq)
            seq_order.sort()
            first_order = seq_order[0]
            last_order = seq_order[-1]
            index = 0
            while first_order + 20 <= last_order:
                for i in range(20):
                    new_path = 'img' + str(int(first_order + i)) + '.jpg'
                    all_path_order.append(new_path)
                video_dir = 'C:\\Users\\PVer\\Desktop\\output\\%s\\%s' % (emotion_list[0], jpg_path.split('\\')[-1]) + '_%s.avi' % (str(index))
                print(video_dir)
                fps = 30
                img_size = (640, 480)

                fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')  # opencv3.0
                videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
                for filename in all_path_order:
                    frame = cv2.imread(os.path.join(jpg_path, filename))
                    videoWriter.write(frame)
                first_order += 10
                index += 1
                all_path_order.clear()

video_crop(im_dir)
print('finish + %s' % im_dir)
