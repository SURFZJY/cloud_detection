from keras.preprocessing.image import load_img , img_to_array
from prettytable import PrettyTable
import numpy as np
import os 

def generate_data(path):
    """Load image data from the root directory.
    
    Load image data from the root directory and convert it 
    to numpy array format, automatically generate data and 
    its corresponding labels.
    
    # Parameters
        path : str
            path of the data directory
            in the format like:
            path/
                dogs/
                    dog001.jpg
                    dog002.jpg
                    ...
                cats/
                    cat001/jpg
                    cat002.jpg
                    ...
                elephants/
                    elephant001.jpg
                    elephant002.jpg
                    ...
    # Returns
        numpy array tuple: (data, labels)
    """
    files = os.listdir(path)
    data = []
    labels = []
    nb_classes = len(files)
    class_name = []
    class_list = []
    
    for i in range(nb_classes):
        sub_path = path + '\\' + files[i]  
        pics = os.listdir(sub_path)
        nb_pics = len(pics)
        file_name = files[i]
        class_name.append(file_name)
        class_list.append(nb_pics)
        for j in range(nb_pics):
            img_path = sub_path + '\\' + pics[j]
            img = load_img(img_path)
            x = img_to_array(img)
            data.append(x)
            labels.append(i)
    
    data = np.array(data)
    labels = np.array(labels)
    print(sum(class_list), "samples in", nb_classes, "categories")
    print("The shape of each sample is", x.shape)
    table = PrettyTable(["Class_name", "Samples_number", "Label"])
    table.align["Class_name"] = "l"
    table.padding_width = 1
    for i in range(nb_classes):
        table.add_row([class_name[i], class_list[i], i])
    print(table)
    print("Generated data_size is", data.shape)
    print("Generated labels_size is", labels.shape)
    return data, labels

if __name__ == "__main__":
    data, labels = generate_data("H:\surfzjy\workspace\keras_study\practise\swimcat_data")