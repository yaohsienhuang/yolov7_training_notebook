import os
import cv2
import shutil
import random
import time
import numpy as np
import xml.etree.ElementTree as ET
from IPython.display import clear_output
from .FileProcessing import fileProcessing
from .ProgressBar import progressBar

class xmlProcessing(fileProcessing):
    def __init__(self,classes_dict=None):
        if classes_dict is None:
            self.classes_dict = {'OK':0, 'NG':1}
            print('classes_dict=\t',self.classes_dict)
            print('the classes_dict can be setup, using func => setup_classes_dict(classes_dict={...})')
    
    def setup_classes_dict(self,classes_dict):
        self.classes_dict=classes_dict
        print('classes_dict=\t',classes_dict)
        
    def extract_both_exist_image_xml(self,path,target,mode,image_extension='JPG'):
        '''sample:
        fileProcessing().extract_both_exist_image_xml(
            path='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model-preparing/FM_OK-0',
            target='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model/FM_OK',
            mode='copy',
        )
        '''
        target=os.path.normpath(target)+os.sep
        xml_list=super().get_file_list(path,['xml'])
        total_n=len(xml_list)*2
        print('xml+image=\t',total_n)
        progress_Bar=progressBar()
        progress_Bar.start(name='extract_both_exist_image_xml')
        cnt=0
        for path in xml_list:
            if not os.path.isdir(target):
                os.makedirs(target,exist_ok=True)
            image_path=path.replace('.xml', f'.{image_extension}')
            if mode=='move':
                shutil.move(path,target)
                shutil.move(image_path,target)
            elif mode=='copy':
                shutil.copy(path,target)
                shutil.copy(image_path,target)
            cnt+=2
            progress_Bar.update(cnt,total_n)
    
    def read_xml_label_counts(self,path):
        xml_list=super().get_file_list(path,['xml'])
        result=[]
        for xml in xml_list:
            tree = ET.parse(xml)
            root = tree.getroot()
            for elem in root.findall('object'):
                name = elem.find('name').text
                result.append(name)
        values, counts = np.unique(result, return_counts=True)
        print('------ the count of labels ------')
        for i in range(len(values)):
            print(f'{values[i]}=\t{counts[i]} labels')
    
    def bbx_to_xml(self,bbx_data,target):

        
    
    def replace_xml_object_name(self,path,replace_dict):
        '''sample:
        xmlProcessing().replace_xml_object_name(
            path='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/FM_model',
            replace_dict={'FM_OK':'OK_Area','FM_NG':'NG_Other'}
        )
        '''
        xml_list=super().get_file_list(path,['xml'])
        total_n=len(xml_list)
        print('xml=\t',total_n)
        key_list=list(replace_dict.keys())
        print('replace_dict=\t',replace_dict)
        
        print('====================== Before ======================')
        self.read_xml_label_counts(path)
        
        for file in xml_list:
            tree = ET.parse(file)
            root = tree.getroot()
            name_elts = root.findall(".//name")    # we find all 'name' elements
            for elt in name_elts:
                for key,values in replace_dict.items():
                    elt.text = elt.text.replace(key,values)
            tree.write(file)
                        
        print('====================== After ======================')
        self.read_xml_label_counts(path)
        
    def move_object_name(self,path,object_name,mode='copy',image_extension='JPG'):
        '''sample:
        xmlProcessing().move_object_name(
            path='/tf/cp1ai01/COG/03_POC訓練資料/object_detection/Diesaw_model/Diesaw_model-checked',
            object_name='NG_Chipping',
            mode='move'
        )
        '''
        xml_list=super().get_file_list(path,['xml'])
        total_n=len(xml_list)
        cnt=0
        for path in xml_list:
            tree = ET.parse(path)
            root = tree.getroot()
            name_elts = root.findall(".//name")
            check=sum([1 for elt in name_elts if elt.text==object_name])
            if check>=1:
                image_path=path.replace('.xml', f'.{image_extension}')
                target=os.path.split(path)[0]+os.sep+object_name+os.sep
                if not os.path.isdir(target):
                    os.makedirs(target,exist_ok=True)
                if mode=='move':
                    shutil.move(path,target)
                    shutil.move(image_path,target)
                elif mode=='copy':
                    shutil.copy(path,target)
                    shutil.copy(image_path,target)
                cnt+=2
            print('Complete(img+xml)=\t',cnt)
            
    def getYoloFormat(self,filename, label_path, img_path, yolo_path, dataset_type):
        '''sample:
        labelpath = 'FM_model-checked/'
        imgpath = 'FM_model-checked/'
        yolopath = 'FM_model-checked_yolov7/'
        train_test_split = 0.1
        os.makedirs(yolopath + 'images/train', exist_ok=True)
        os.makedirs(yolopath + 'images/val', exist_ok=True)
        os.makedirs(yolopath + 'labels/train', exist_ok=True)
        os.makedirs(yolopath + 'labels/val', exist_ok=True)
        xml_list = sorted([file for file in os.listdir(labelpath) if file.split('.')[-1] == 'xml'])
        random.shuffle(xml_list)
        total_progress = len(xml_list)
        progress = 0
        progress_Bar=progressBar()
        progress_Bar.start('getYoloFormat')
        xml_handler=xmlProcessing().setup_classes_dict(classes_dict={'OK_Area':0,'NG_Area':1,'NG_Other':2})
        for f in xml_list:
            progress += 1
            try:
                if progress/len(xml_list) > train_test_split:
                    xml_handler.getYoloFormat(f, labelpath, imgpath, yolopath, 'train')
                else:
                    xml_handler.getYoloFormat(f, labelpath, imgpath, yolopath, 'val')
            except Exception as e:
                print(e)

            progress_Bar.update(progress,total_progress)
        '''
        imgname = filename.replace('.xml', '.JPG')
        tree = ET.parse(label_path+filename)
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
            ary.append(' '.join([str(self.classes_dict[objclass]), str(x), str(y), str(w), str(h)]))

        if os.path.exists(img_path+imgname):                              # 圖片本來在image裡面，把圖片移到yolo資料夾下    
            shutil.copyfile(img_path+imgname, yolo_path + 'images/' + dataset_type + '/' + imgname)     #同時把yolo參數檔寫到yolo之下
            with open(yolo_path + 'labels/' + dataset_type + '/' + imgname.replace('.JPG', '.txt'), 'w') as f:
                f.write('\n'.join(ary))
            with open(yolo_path + dataset_type + '_list.txt', 'a') as f:
                f.write(yolo_path + 'images/' + dataset_type + '/' + imgname + '\n')
        else:
            print(img_path+imgname, 'not found')

    class save_xml():
        def __init__(self,data,target):
            self.export(data,target)
        
        def export(self,data,target):
            for result in data:
                annotation = ET.Element('annotation')
                annotation.text = '\n\t'
                root = ET.ElementTree(annotation)
                self.imginfo(annotation, result['file_name'],result['size_width_height'][0],result['size_width_height'][1],'AI',' ')
                if result['bbox'] is not None:
                    for item in result['bbox']:
                        self.bboxinfo(annotation, item[4],item[5],item[0],item[1], item[2], item[3], end = False)

                save_xml_file_name = os.path.join(target, '.'.join(os.path.basename(result['file_name']).split('.')[0:-1]))
                root.write(save_xml_file_name + '.xml', encoding ='utf-8')
                
        def createlayer(self,annotation, tag, content, layer, end = False):
            sub = ET.SubElement(annotation, tag)
            sub.text = content
            sub.tail = '\n'
            if end:
                tabnum = layer - 1
            else:
                tabnum = layer    
            for i in range(tabnum):
                sub.tail += '\t'      
            return sub
        
        def imginfo(self,annotation, path, imgwidth, imgheight,editor,classes):
            patharr = path.split('\\')
            numpath = len(patharr) - 1
            createlayer(annotation, 'folder', patharr[numpath -1], 1)    
            createlayer(annotation, 'filename', patharr[numpath], 1)
            createlayer(annotation, 'path', path, 1)
            source = createlayer(annotation, 'source', '\n\t\t', 1)
            createlayer(source, 'database', 'Unknown', 2,  end = True)
            size = createlayer(annotation, 'size', '\n\t\t', 1)
            createlayer(size, 'width', str(imgwidth), 2)
            createlayer(size, 'height', str(imgheight), 2)
            createlayer(size, 'depth', '1', 2, end = True)
            createlayer(annotation, 'editor', editor, 1)
            createlayer(annotation, 'classes', classes, 1)
            createlayer(annotation, 'segmented', '0', 1)

        def bboxinfo(self,annotation, label, score, xmin, ymin, xmax, ymax, end = False):
            if end:
                object = createlayer(annotation, 'object', '\n\t\t', 1, end = True)
            else:
                object = createlayer(annotation, 'object', '\n\t\t', 1)
            createlayer(object, 'name', label, 2)
            createlayer(object, 'score', str(score), 2)
            createlayer(object, 'pose', 'Unspecified', 2)
            createlayer(object, 'truncated', '0', 2)
            createlayer(object, 'difficult', '0', 2)
            bndbox = createlayer(object, 'bndbox', '\n\t\t\t', 2, end = True)
            createlayer(bndbox, 'xmin', str(xmin), 3)
            createlayer(bndbox, 'ymin', str(ymin), 3)
            createlayer(bndbox, 'xmax', str(xmax), 3)
            createlayer(bndbox, 'ymax', str(ymax), 3, end = True)
            
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
    progress_Bar=progressBar()
    progress_Bar.start()
    for f in xml_list:   #透過getYoloFormat將圖像和參數檔全部寫到YOLO下
        progress += 1
        try:
            if progress/len(xml_list) > train_test_split:
                XMLprocessing.getYoloFormat(f, labelpath, imgpath, yolopath, 'train')
            else:
                XMLprocessing.getYoloFormat(f, labelpath, imgpath, yolopath, 'val')
        except Exception as e:
            print(e)
        progress_Bar.update(progress,total_progress)
