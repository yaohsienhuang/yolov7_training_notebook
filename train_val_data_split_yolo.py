#!/usr/bin/env python
# coding: utf-8

# In[6]:


import xml.etree.ElementTree as ET
import os
import cv2
import shutil
import random
from IPython.display import clear_output

status_dic = {'OK_Area':0,'NG_Area':1,'NG_Other':2}             #用dictionary 記錄label的名稱

def getYoloFormat(filename, label_path, img_path, yolo_path, dataset_type):    
    imgname = filename.replace('.xml', '.JPG')
    tree = ET.parse(label_path+filename)#讀取xml
    root = tree.getroot()
    image_h, image_w, _ = cv2.imread(img_path+imgname).shape
    ary = []
    for object in root.findall('object'):
        objclass = object.find('name').text
        obj = object.find('bndbox')
        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)
        xmin_ = None
        ymin_ = None
        # 如果大小相反了，轉換一下
        if xmin > xmax:
            xmin_ = xmax
            xmax = xmin
            xmin = xmin_
        if ymin > ymax:
            ymin_ = ymax
            ymax = ymin
            ymin = ymin_

        x = (xmin + (xmax-xmin)/2) * 1.0 / float(image_w)    #YOLO吃的參數檔有固定的格式
        y = (ymin + (ymax-ymin)/2) * 1.0 / float(image_h)    #先照YOLO的格式訂好x,y,w,h
        w = (xmax-xmin) * 1.0 / float(image_w)
        h = (ymax-ymin) * 1.0 / float(image_h)
        if x > 1 or x < 0 or y > 1 or y < 0 or w > 1 or w < 0 or h > 1 or h < 0:
            print(yolo_path + 'labels/' + dataset_type + '/' + imgname.replace('.JPG', '.txt'))
            print(image_w,image_h,xmin,xmax,ymin,ymax,x,y,w,h)
        ary.append(' '.join([str(status_dic[objclass]), str(x), str(y), str(w), str(h)]))

    if os.path.exists(img_path+imgname):                              # 圖片本來在image裡面，把圖片移到yolo資料夾下    
        shutil.copyfile(img_path+imgname, yolo_path + 'images/' + dataset_type + '/' + imgname)     #同時把yolo參數檔寫到yolo之下
        with open(yolo_path + 'labels/' + dataset_type + '/' + imgname.replace('.JPG', '.txt'), 'w') as f:
            f.write('\n'.join(ary))
        with open(yolo_path + dataset_type + '_list.txt', 'a') as f:
            f.write(yolo_path + 'images/' + dataset_type + '/' + imgname + '\n')
    else:
        print(img_path+imgname, 'not found')
                
# In[ ]:


if __name__ == '__main__':
    labelpath = 'FM_model/'           #設定路徑
    imgpath = 'FM_model/'
    yolopath = 'FM_model_yolov7/'
    train_test_split = 0.1
    os.makedirs(yolopath + 'images/train', exist_ok=True)
    os.makedirs(yolopath + 'images/val', exist_ok=True)
    os.makedirs(yolopath + 'labels/train', exist_ok=True)
    os.makedirs(yolopath + 'labels/val', exist_ok=True)
    xml_list = sorted([file for file in os.listdir(labelpath) if file.split('.')[-1] == 'xml'])
    random.shuffle(xml_list)
    total_progress = len(xml_list)
    progress = 0
    for f in xml_list:   #透過getYoloFormat將圖像和參數檔全部寫到YOLO下
        progress += 1
        try:
            if progress/len(xml_list) > train_test_split:
                getYoloFormat(f, labelpath, imgpath, yolopath, 'train')
            else:
                getYoloFormat(f, labelpath, imgpath, yolopath, 'val')
        except Exception as e:
            print(e)

