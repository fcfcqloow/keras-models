import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.util import loadCacheFunction, loadCacheVariable
from src.face_gender_age import mat

sys.modules['mat'] = mat


cascade = cv2 .CascadeClassifier("./definition/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("./definition/haarcascade_eye.xml")

def train(args):
    with tf.device("/gpu:0"):
        model, history = train()
        model.save('model.h5')

def train():
    gender_range = 2
    age_range    = 100
    batch_size   = 128
    epochs       = 30
    image_size   = 64
    loss         = "categorical_crossentropy"
    optimizer    = Adam(learning_rate=0.001)

    def build_model(
        x, 
        optimizer=Adam(learning_rate=0.001), 
        loss="categorical_crossentropy",
    ) -> Model:
        base_model = EfficientNetB3(
            weights = "imagenet",
            include_top=False,
            input_shape=x.shape[1:],
        )
        base_model.trainable = False
        
        gender_model = base_model.output
        gender_model = GlobalAveragePooling2D()(gender_model)
        gender_model = Dense(1024,activation='relu')(gender_model)
        gender_model = Dense(gender_range, activation="softmax", name="gender")(gender_model)
        
        age_model = base_model.output
        age_model = GlobalAveragePooling2D()(age_model)
        age_model = Dense(1024,activation='relu')(age_model)
        age_model = Dense(age_range, activation="softmax", name="age")(age_model)
        
        model = Model(
            inputs=base_model.input,
            outputs=[
                gender_model,
                age_model,
            ]
        )

        for i in model.layers:
            print(i.name, i.trainable)

        model.compile(
            optimizer=optimizer,
            loss={
                "age"    : loss,
                "gender" : loss,
            },
            metrics=["accuracy"],
        )
        return model
    def createVariables():
        x, age, gender = [], [], []
        files = loadCacheFunction('cache/face_file.b', mat.getFaceFilePaths)
        
        X_train, sX_train           = loadCacheVariable('cache/v_X_train')
        X_test, sX_test             = loadCacheVariable('cache/v_X_test')
        age_train, sage_train       = loadCacheVariable('cache/v_age_train')
        age_test, sage_test         = loadCacheVariable('cache/v_age_test')
        gender_train, sgender_train = loadCacheVariable('cache/v_gender_train')
        gender_test, sgender_test   = loadCacheVariable('cache/v_gender_test')
        
       
        
        if X_train is not None:
            return np.array(X_train), np.array(X_test), np.array(age_train), np.array(age_test), np.array(gender_train), np.array(gender_test)
        for i, file in enumerate(files):
            if int(file.age) >= age_range:
                continue
            if i % 100 == 0:
                print(i, "...")
            img = cv2.imread(file.path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_list = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            for _x, _y, weight, height in face_list:
                try:
                    img = img[_y:_y+height, _x:_x+weight]
                    eyes = eye_cascade.detectMultiScale(img)
                    if len(eyes) != 2:
                        continue
                    img = cv2.resize(img, (image_size, image_size))
                    # print(file.path, '.   /gray/%s_%s_%s.jpg' % (i, file.gender, file.age))
                    # try:
                    cv2.imwrite('./gray/%s_%s_%s.jpg' % (i, file.gender, file.age), img)
                    # except:
                    #     print('cant imwrite', file.path)
                    img = img[...,::-1]
                    arr = img_to_array(img)
                    x.append(arr)
                    age.append(int(file.age))
                    gender.append(0 if file.gender == 'male' else 1)
                except:
                    print('err', file.path)

        age, gender = to_categorical(age, age_range), to_categorical(gender, gender_range)

        X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(x, age, gender, test_size=0.25, random_state=111)
        X_train      = sX_train(X_train)
        X_test       = sX_test(X_test)
        age_train    = sage_train(age_train) 
        age_test     = sage_test(age_test)
        gender_train = sgender_train(gender_train)
        gender_test  = sgender_test(gender_test)

        return np.array(X_train),  np.array(X_test),  np.array(age_train),  np.array(age_test),  np.array(gender_train),  np.array(gender_test)
    
    # X_train, X_test, age_train, age_test, gender_train, gender_test = loadCacheVariable('cache/variables.b', createVariables)
    X_train, X_test, age_train, age_test, gender_train, gender_test = createVariables()
    print(X_train.shape, X_test.shape, age_train.shape, age_test.shape, gender_train.shape, gender_test.shape)
    model = build_model(X_train, optimizer=optimizer, loss=loss)
    model.summary()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.0001
    )
    history = model.fit(
        X_train, {
            "age"    : age_train,
            "gender" : gender_train,
        },
        batch_size=batch_size,
        epochs=epochs,
        validation_data = (X_test, {
            "age"    : age_test,
            "gender" : gender_test,
        }),
        # callbacks=[early_stopping, reduce_lr],
    )
    model.evaluate(X_test, {
        "age"    : age_test,
        "gender" : gender_test,
    })
    
    plt.plot(history.history['gender_accuracy'])
    plt.plot(history.history['age_accuracy'])
    plt.plot(history.history['val_gender_accuracy'])
    plt.plot(history.history['val_age_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['gender_accuracy', 'age_accuracy', 'val_gender_accuracy', 'val_age_accuracy'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return model, history
