import cv2
import os 
import numpy as np

def cut_img(path, ROI):
    """Cut image from a single directory.
    
    Cut image from a single directory and automatically 
    create a new directory called 'path_cut' in the upper 
    directory of the path to store the new cutted images.
    
    # Parameters
        path : str
            path of the data directory in the format like:
            path/
                img001.jpg
                img002.jpg
                img003.jpg
                img004.jpg
        ROI : list or tuple
            region of interest, in the order like:
            (row_start, row_end, column_start, column_end)
    """   
    files = os.listdir(path)
    
    path_item = path.split('\\')
    parent_dir = path_item[0]
    for i in path_item[1:-1]:
        parent_dir += '\\'
        parent_dir += i
    save_dir = parent_dir + '\\' + path_item[-1] + '_cut\\'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in files:
        img = cv2.imread(path+ '\\' +i)
        content = img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
        new_name = save_dir + i
        cv2.imwrite(new_name, content)
    
    return True

if __name__ == "__main__":
    path = 'H:\surfzjy\wuxi_20150211'
    ROI = (1000,3320,260,2580)
    cut_img(path, ROI)
    print('Finish~')