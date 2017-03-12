import cv2
import os 
import numpy as np

path = 'H:\surfzjy\Cloud_observation\data\wholesky'
files = os.listdir(path)

path_list = []
nb_classes = len(files)

i = 6
sub_path = path + '\\' + files[i]
pics = os.listdir(sub_path)
nb_pics = len(pics)
for j in pics:
    img_path = sub_path + '\\' + j
    if j!='Thumbs.db':
        path_list.append(img_path)
        
for x in range(len(path_list)):
    img = cv2.imread(path_list[x])
    mask = cv2.imread('mask.png', 0)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    resized_img = cv2.resize(masked_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    new_name = pics[x]
    cv2.imwrite(new_name, resized_img)
#     cv2.imwrite(new_name, masked_img)