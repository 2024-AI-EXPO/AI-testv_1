import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
import cv2
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization,Activation,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.keras.applications import VGG16
from keras.models import Model
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall
from keras.metrics import Recall
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




#C:/Users/modeep1/Downloads/archive/asl_alphabet_train/asl_alphabet_train
#C:/Users/modeep1/Downloads/archive/asl_alphabet_test/asl_alphabet_test
train_dir = 'C:/Users/modeep1/Downloads/archive/asl_alphabet_train/asl_alphabet_train'
test_dir = 'C:/Users/modeep1/Downloads/archive/asl_alphabet_test/asl_alphabet_test'
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
               'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
               'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}

def load_train_data():
    Y_train = []
    X_train = []
    size = 64, 64
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
            # read image
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image,0)
            # resize image
            temp_img = cv2.resize(temp_img, size)
            #load converted classes
            Y_train.append(labels_dict[folder])
            X_train.append(temp_img)
    #convert X_train to numpy
    X_train = np.array(X_train)
    #normalize pixels of X_train
    X_train = X_train.astype('float32')/255.0
    #convert from 1-channel to 3-channel
    X_train = np.stack((X_train,)*3, axis=-1)
    #convert Y_train to numpy
    Y_train = np.array(Y_train)
    print()
    print("Pixels after normalize : min = %d max = %d "%(X_train.min(),X_train.max()))
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)

    return X_train, Y_train


def load_test_data():
    labels = []
    X_test = []
    size = 64, 64
    for image in os.listdir(test_dir):
        # read image
        temp_img = cv2.imread(test_dir + '/'+ image,0)
        # resize image
        temp_img = cv2.resize(temp_img, size)
        # load converted classes
        labels.append(labels_dict[image.split('_')[0]])
        X_test.append(temp_img)
    #convert X_test to numpy
    X_test = np.array(X_test)
    #normalize pixels of X_test
    X_test = X_test.astype('float32')/255.0
    #convert from 1-channel to 3-channel in Gray
    X_test = np.stack((X_test,)*3, axis=-1)
    #convert Y_test to numpy
    Y_test = np.array(labels)
    print("Pixels after normalize : min = %d max = %d "%(X_test.min(),X_test.max()))
    print('Loaded', len(X_test), 'images for testing,', 'Test data shape =', X_test.shape)

    return X_test, Y_test

X_train, Y_train = load_train_data()
X_test , Y_test = load_test_data()

print("Y_train befor one-hot encoder : ",Y_train[0])
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print("Y_train befor one-hot encoder : ",Y_train[0])

print('Loaded', len(Y_test), 'images for testing,', 'Test data shape =', Y_test.shape)




datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False )
datagen.fit(X_train)


model = Sequential()

model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3)))

model.add(Flatten())

model.add(Dense(420, activation='relu')) 

model.add(Dense(29, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[AUC(name = 'Accuracy'),Precision(name = 'Precision'),Recall(name = 'Recall')] )#optimizer = sigmoed

model.summary()

tf.config.run_functions_eagerly(True)
history = model.fit(
    datagen.flow(X_train,Y_train, batch_size = 128),
    epochs = 1,
    validation_data = (X_test, Y_test),
    callbacks = [
        ModelCheckpoint('C:/Users/modeep1/Desktop/github/AI-testv_1/model.keras', verbose=1, save_best_only=True),
        ReduceLROnPlateau()
    ]
)
