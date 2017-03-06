from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
import numpy as np
import os 
import random

path = 'H:\surfzjy\workspace\keras_study\practise\swimcat_data'
files = os.listdir(path)
path_list = []
labels = []
nb_classes = len(files)

for i in range(nb_classes):
    sub_path = path + '\\' + files[i]
    pics = os.listdir(sub_path)
    nb_pics = len(pics)
    for j in range(nb_pics):
        img_path = sub_path + '\\' + pics[j]
        path_list.append(img_path)
        labels.append(i)

tmp = list(zip(path_list, labels))
random.shuffle(tmp)
path_list, labels = zip(*tmp)

test_ratio = 0.2
nb_data = len(labels)
nb_test = round(nb_data * test_ratio)
test_list = path_list[:nb_test]
test_label = labels[:nb_test]
train_list = path_list[nb_test:]
train_label = labels[nb_test:]

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(125,125,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',metrics=['accuracy'])

label_dock = []
data_dock = []
dock_thres = 60
dock_init = 0
dock_cot = 0 

def model_fit(model ,data_dock, label_dock):

    data_array = np.array(data_dock)
    label_array = np.array(label_dock)

    data_array /= 255

    label_array = np_utils.to_categorical(label_array, nb_classes=5)

    model.fit(data_array, label_array, 
              nb_epoch=50, batch_size=20,
              verbose=1)    

for i in range(len(train_list)):

    img = load_img(train_list[i])
    x = img_to_array(img)
    data_dock.append(x)
    label_dock.append(train_label[i])
    dock_init += 1
#     print(dock_init)

    if dock_init >= dock_thres:
#         print('Team', dock_cot+1)
        model_fit(model, data_dock, label_dock)
        dock_cot += 1
        data_dock = []
        label_dock = []
        dock_init = 0

##============== Test ======================
test_data = []
test_labels = []
test_scores = []
for i in range(len(test_list)):
    img = load_img(test_list[i])
    x = img_to_array(img)
    test_data.append(x)
    test_data = np.array(test_data)
    test_data /= 255
    test_labels.append(test_label[i])
    test_labels = np_utils.to_categorical(test_labels, nb_classes=5)
#     print('test data:', test_data)
#     print('tset_data.shape:', test_data.shape)
#     print('test labels:', test_labels)
    score = model.evaluate(test_data, test_labels, verbose=0)
    test_scores.append(score[1])
    test_data = []
    test_labels = []
#     print('Test accuracy:', score[1])
    
test_scores = np.array(test_scores)
# print('Test scores:', test_scores)
print('Test Accuracy:', test_scores.mean())
