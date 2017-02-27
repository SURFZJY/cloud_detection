from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing.image import load_img , img_to_array
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.applications.vgg16 import VGG16, preprocess_input
from surfola import generate_data
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from prettytable import PrettyTable

import numpy as np
import os 
        
if __name__ == "__main__":
    data_path = "H:\surfzjy\workspace\keras_study\practise\swimcat_data"
    data, labels = generate_data(data_path)
    data /= 255
### ===================================
    # Testing data ratio
    test_size_ratio=0.20
    # epoch of each iteration
    nb_epoch = 50
    # Repeat the experiment n times
    n_times = 50
### ===================================
    test_sum_score = 0.0
    cot = 1

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size_ratio)
    train_labels = np_utils.to_categorical(train_labels, nb_classes=5)
    test_labels = np_utils.to_categorical(test_labels, nb_classes=5)   
    
    
    model = VGG16(weights='imagenet', include_top=False)
    
    train_bnfeature = model.predict(train_data)
    test_bnfeature = model.predict(test_data)
    np.save(open('train_bnfeature.npy', 'wb'), train_bnfeature)
    np.save(open('test_bnfeature.npy', 'wb'), test_bnfeature)
     
    train_data = np.load(open('train_bnfeature.npy','rb'))
    test_data = np.load(open('test_bnfeature.npy','rb'))

    while(cot <= n_times):
            
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        csv_logger = CSVLogger('epoch' + str(nb_epoch) + '_' + str(cot) + '.csv')
            
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=16,
                  verbose=1, 
                  validation_data=None,
                  shuffle=True,
                  callbacks=[csv_logger])
        score = model.evaluate(test_data, test_labels, verbose=0)
        print('Test accuracy:', score[1])
        test_sum_score += score[1]
        with open('H:/surfzjy/cloud_detection/epoch'+str(nb_epoch)+'.txt', 'a+') as f:
            f.write('Round ' + str(cot) + ': Test accuracy: ' + str(score[1]))
            f.write('\n')                 
        cot += 1
    test_ave_score = test_sum_score / n_times
    with open('H:/surfzjy/cloud_detection/epoch'+str(nb_epoch)+'.txt', 'a+') as f:
        f.write("Average Test Accuracy : " + str(test_ave_score))
    f.close()