from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
# from keras.preprocessing.image import load_img , img_to_array
from keras.callbacks import CSVLogger
from keras.utils import np_utils
# from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
# from prettytable import PrettyTable
from surfola import generate_data
import numpy as np
import os 
# import cv2

def conv_model():
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 125, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
#     model.add(Convolution2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     
#     model.add(Convolution2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['accuracy'])
        
    return model

data_path = 'H:\surfzjy\SWIMCAT'
data, labels = generate_data(data_path)
data /= 255

test_size_ratio = 0.20

validation_switch = True  
n_folds = 5
nb_epoch = 50
n_times = 5
# test_sum_score = 0.0 
cot = 1

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size_ratio)
np.save(open('SWIMCAT_train_data.npy', 'wb'), train_data)
np.save(open('SWIMCAT_test_data.npy', 'wb'), test_data)
np.save(open('SWIMCAT_train_labels.npy', 'wb'), train_labels)
np.save(open('SWIMCAT_test_labels.npy', 'wb'), test_labels)

# train_data = np.load(open('SWIMCAT_train_data.npy', 'rb'))
# test_data = np.load(open('SWIMCAT_test_data.npy', 'rb'))
# train_labels = np.load(open('SWIMCAT_train_labels.npy', 'rb'))
# test_labels = np.load(open('SWIMCAT_test_labels.npy', 'rb'))

train_labels = np_utils.to_categorical(train_labels, nb_classes=5)
test_labels = np_utils.to_categorical(test_labels, nb_classes=5) 

model = conv_model()

while(cot <= n_times):
    if validation_switch:
        val_cot = 1
        kf = KFold(train_data.shape[0], n_folds)
        for train_index, val_index in kf:
             X_train, X_val = train_data[train_index], train_data[val_index]
             y_train, y_val = train_labels[train_index], train_labels[val_index]
             csv_logger = CSVLogger('epoch' + str(nb_epoch) + '_' + str(cot) + '_val_' + str(val_cot) + '.csv')
             val_cot += 1
             model = conv_model()
             model.fit(X_train, y_train,  
                       nb_epoch=nb_epoch, batch_size=16,
                       verbose=1, 
                       validation_data=(X_val, y_val),
                       callbacks=[csv_logger])  
    cot += 1

# csv_logger = CSVLogger('epoch' + str(nb_epoch) + '_' + str(cot) + '.csv')
# model = conv_model()
# model.fit(train_data, train_labels,
#           nb_epoch=nb_epoch, batch_size=16,
#           verbose=1, 
#           shuffle=True,
#           validation_data=None,
#           callbacks=[csv_logger])

# model.save('CAMS_model.h5')

# score = model.evaluate(test_data, test_labels, batch_size=16, verbose=0)
# print('Test accuracy:', score[1])
#   test_sum_score += score[1]
# with open('H:/surfzjy/cloud_detection/epoch'+str(nb_epoch)+'.txt', 'a+') as f:
#     f.write('Round ' + str(cot) + ': Test accuracy: ' + str(score[1]))
#     f.write('\n')

# test_ave_score = test_sum_score / n_times
# with open('H:/surfzjy/cloud_detection/epoch'+str(nb_epoch)+'.txt', 'a+') as f:
#     f.write("Average Test Accuracy : " + str(test_ave_score))
# f.close()