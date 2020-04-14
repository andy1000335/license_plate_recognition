import numpy as np
import os
from PIL import Image

folder = r'\data'
def get_path(folder):
    path = []
    for filename in sorted(os.listdir(folder)):
        name = os.path.join(folder, filename)
        path.append(name)
    return path

def img2array(datas):
    all_array = []
    for data in datas:
        img = Image.open(data)
        array = np.array(img)
        array = array
        all_array.append(array)
    return all_array

def shrink_img(all_img):
    ShrinkImg=[]
    for i in range(len(all_img[:])):
        new_img=[]
        for row in range(len(all_img[i][:])):
            if len(all_img[i][:])/4 < row < len(all_img[i][:])*3/4:
                img_col=[]
                for col in range(len(all_img[i][row][:])):
                    if len(all_img[i][row][:])/3 < col < len(all_img[i][row][:])*2/3:
                        img_col.append(all_img[i][row][col])
                new_img.append(img_col)
        ShrinkImg.append(new_img)
    return np.array(ShrinkImg)

def get_label(data_pathes):
    labels = []
    for data_path in data_pathes:
        name = data_path[len(folder)+1:len(folder)+7]
        labels.append(name)
    return np.array(labels)

all_DataPath = get_path(folder)
a=img2array(all_DataPath)
a=np.array(a)

small=shrink_img(a)
all_label=get_label(all_DataPath)
print(small[0].shape)
print(all_label.shape)
print(all_label[0])

for i in range(small.shape[0]):
    img=Image.fromarray(np.uint8(small[i]))
    path=('/data/shrink_img/'+all_label[i]+'.png')
    img.save(path)