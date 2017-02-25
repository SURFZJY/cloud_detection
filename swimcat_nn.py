from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing.image import load_img , img_to_array
from keras.callbacks import CSVLogger
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from prettytable import PrettyTable
import numpy as np
import os 

def generate_data(path):
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
    data, labels = generate_data(data_path)
    data /= 255
    # Testing data ratio
    test_size_ratio=0.20
    # Validation folds number
    # There is some problem with the validation method, so set it False
    validation_switch = False
    n_folds = 5

    # epoch of each iteration
    nb_epoch = 3

    test_sum_score = 0.0
    
    cot = 1
        
    # Repeat the experiment n times
    n_times = 5
    
    while(cot <= n_times):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size_ratio)
        train_labels = np_utils.to_categorical(train_labels, nb_classes=5)
        test_labels = np_utils.to_categorical(test_labels, nb_classes=5)

        model = conv_model(loss='categorical_crossentropy', 
                           optimizer='rmsprop', metrics=['accuracy'])

        if validation_switch:
            val_cot = 1
            kf = KFold(train_data.shape[0], n_folds)
            
            for train_index, val_index in kf:
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_labels[train_index], train_labels[val_index]
                csv_logger = CSVLogger('epoch' + str(nb_epoch) + '_' + str(cot) + '_val_' + str(val_cot) + '.csv')
                val_cot += 1
                model.fit(X_train, y_train,
                          nb_epoch=nb_epoch, batch_size=16,
                          verbose=1, 
                          validation_data=(X_val, y_val),
                          callbacks=[csv_logger])  

        csv_logger = CSVLogger('epoch' + str(nb_epoch) + '_' + str(cot) + '.csv')
            
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=16,
                  verbose=1, 
                  validation_data=None,
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