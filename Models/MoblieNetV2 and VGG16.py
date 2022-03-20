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



def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224, 224))
    return image[...,::-1]



datasettrain_path = "./data"
#（1）对每一个批次的训练图片，适时地进行数据增强处理（data augmentation），可以添加其他参数
#（2）图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；
data_with_aug = ImageDataGenerator(horizontal_flip=True, #随机水平翻转
                                   vertical_flip=False,  #随机垂直翻转
                                   rescale=1./255,
                                   #rescale: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，
                                   # 否则将数据乘以所提供的值（在应用任何其他转换之前）。
                                   # rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，
                                  # 在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，
                                  # 因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。
                                validation_split=0.2)  #保留用于验证的图像的比例

train = data_with_aug.flow_from_directory(datasettrain_path,
                                          #directory: 目标目录的路径。每个类应该包含一个子目录。
                                          # 任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。
                                          class_mode="binary",#"binary" 将是 返回1D 二进制标签
                                          target_size=(96, 96),  #所有的图像将被调整到的尺寸
                                          batch_size=32,#一批数据的大小（默认 32）
                                          subset="training")#数据子集 ("training" 或 "validation")

val = data_with_aug.flow_from_directory(datasettrain_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="validation"
                                          )


mnet = MobileNetV2(include_top = False, weights = "imagenet" ,input_shape=(96,96,3))

#两种模型结合
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(96,96,3))


model_1 = Model(inputs=mnet.input, outputs=mnet.get_layer('out_relu').output)
model_2 = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('block5_pool').output)


inp=Input((96,96,3))
r1 = model_1(inp)
r2 = model_2(inp)

x = concatenate([r1,r2], axis=-1)

merge_model = Model(inputs=inp, outputs=x)
merge_model.summary()

tf.keras.backend.clear_session()
model = Sequential([merge_model,
                    GlobalAveragePooling2D(),
                    #全局平均池化，深度神经网络中经常使用的一个层，使用前后的尺寸分别为[B,H,W,C]->[B,C].
                    #特别要注意，这个层使用之后特征图尺寸的维度变成了2维而非4维。这将对你之后的计算带来影响
                    Dense(512, activation = "relu"),
                    BatchNormalization(),
                    #解决传统的神经网络训练需要我们人为的去选择参数，
                    # 比如学习率、参数初始化、权重衰减系数、Drop out比例 的问题，并能提高算法的收敛速度
                    Dropout(0.3),
                    Dense(128, activation = "relu"),
                    Dropout(0.1),
                    Dense(32, activation = "relu"),
                    Dropout(0.3),
                    Dense(2, activation = "softmax")])

 #设置网络模型不可训练,冻层操作 原因：训练自己的分类器时，是不能将预训练网络一起进行训练的，否则就会破坏预训练网络的原始参数。
model.layers[0].trainable = False
#对网络模型进行配置
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.summary()

#schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001
#该回调函数是用于动态设置学习率
lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

hist = model.fit(train,
                    epochs=20,
                    callbacks=[lr_callbacks],
                    validation_data=val)
#绘制了训练和验证精度/损失的学习曲线。
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

#验证集验证
predictions = model.predict(val)
val_path = "../data/"
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