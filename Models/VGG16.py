import numpy as np
from keras import Model, Input
from keras.backend import concatenate
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,BatchNormalization, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from Models.MoblieNetV2 import data_with_aug, lr_callbacks


def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224, 224))
    return image[...,::-1]

datasettrain_path = "./data"
#第二中模型，VGG16模型
train = data_with_aug.flow_from_directory(datasettrain_path,
                                          class_mode="binary",
                                          target_size=(224, 224),
                                          batch_size=98,
                                          subset="training")

val = data_with_aug.flow_from_directory(datasettrain_path,
                                          class_mode="binary",
                                          target_size=(224, 224),
                                          batch_size=98,
                                          subset="validation"
                                          )



#迁移学习VGG16
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
vgg16_model.output[-1]


model = Sequential([vgg16_model,
                    Flatten(),  #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡

                                        Dense(512, activation = "relu"),
                                        BatchNormalization(),
                                        Dropout(0.3),
                                        Dense(128, activation = "relu"),
                                        Dropout(0.1),
                                        Dense(32, activation = "relu"),
                                        Dropout(0.3),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.summary()
hist =  model.fit_generator(train,
                    epochs=20,
                    callbacks=[lr_callbacks],
                    validation_data=val)
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()




#
# #Creating an array of predicted test images
# #可以建立测试集或者写个前端进行测试，待完善。
predictions = model.predict(val)
val_path = "./data/"

plt.figure(figsize=(15, 15))

start_index = 3
#输出16个图像的预测结果
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    preds = np.argmax(predictions[[start_index + i]])

    gt = val.filenames[start_index + i][0:4]
    if gt == "fake":
        gt = 0
    else:
        gt = 1

    if preds != gt:
        col = "r"
    else:
        col = "g"

    plt.xlabel('i={}, pred={}, gt={}'.format(start_index + i, preds, gt), color=col)
    plt.imshow(load_img(val_path + val.filenames[start_index + i]))
    plt.tight_layout()

plt.show()