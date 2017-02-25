from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing.image import load_img , img_to_array
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
import numpy as np
import os 

def generate_data(path, norm=False):
    """Load image data from the root directory.
    
    Load image data from the root directory and convert it 
    to numpy array format, automatically generate data and 
    its corresponding labels.
    
    # Parameters
        path : str
            path of the data directory
        norm : boolean
            image pixel data divided by 255 or not,
            default value is False.
            (in some experiment, I found normalization can make results better)
            
    # Returns
        numpy array tuple: (data, labels)
    """
    files = os.listdir(path)
    data = []
    labels = []
    for i in range(len(files)):
        sub_path = data_path + '\\' + files[i]  
        pics = os.listdir(sub_path)
        nb_pics = len(pics)
        for j in range(nb_pics):
            img_path = sub_path + '\\' + pics[j]
            img = load_img(img_path)
            x = img_to_array(img)
            data.append(x)
            labels.append(i)
    data = np.array(data)
    labels = np.array(labels)
    if norm:
        data /= 255
    return data, labels

def conv_model(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy']):
    """ Construct my convolutional neural network model and compile it.
    
    # Parameters
        optimizer: str 
            (name of optimizer) or optimizer object.
            default value is "categorical_crossentropy"
        loss: str 
            (name of objective function) or objective function.
            default value is "rmsprop"
        metrics: list 
            list of metrics to be evaluated by the model
            during training and testing.
            Typically you will use `metrics=['accuracy']`.
            default value is ['accuracy']
 
    # Returns
        a compiled model (based on keras)
    
    """
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

    model.compile(loss=loss,
              optimizer=optimizer, 
              metrics=metrics)
        
    return model
        
if __name__ == "__main__":
    data_path = "H:\surfzjy\workspace\keras_study\practise\swimcat_data"
    data, labels = generate_data(data_path, norm=True)
    
    # Testing data ratio
    test_size_ratio=0.20
    # Validation folds number
    # There is some problem with the validation method, so set it False
    validation_switch = False
    n_folds = 5

    cot = 1
    while(cot <= 50):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size_ratio)
        train_labels = np_utils.to_categorical(train_labels, nb_classes=5)
        test_labels = np_utils.to_categorical(test_labels, nb_classes=5)

        model = conv_model(loss='categorical_crossentropy', 
                           optimizer='rmsprop', metrics=['accuracy'])
        # Log file name
        csv_logger = CSVLogger('epoch100_'+str(cot)+'.csv')
        if validation_switch:
            kf = KFold(n, n_folds)
            for train_index, val_index in kf:
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_labels[train_index], train_labels[val_index]
            validation_set = (X_val, y_val)
        else:
            X_train = train_data
            y_train = train_labels
            validation_set = None
    
        model.fit(X_train, y_train,
              nb_epoch=100, batch_size=16,
              verbose=1, 
              validation_data=validation_set,
              callbacks=[csv_logger])
        score = model.evaluate(test_data, test_labels, verbose=0)
        print('Test accuracy:', score[1])
        with open('H:/surfzjy/workspace/keras_study/practise/epoch100.txt', 'a+') as f:
            f.write('Round ' + str(cot) +':  ' + 'Test accuracy: ' + str(score[1]))
            f.write('\n')
        cot += 1
    f.close()