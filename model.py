import tensorflow as tf
from tensorflow.keras.applications import ResNet101V2,VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
def resnet101v2_model(num_classes=5):
    model = ResNet101V2(input_shape=(100,100,3),include_top=False)
   # print(model.layers)
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    x=Flatten()(model.layers[-1].output)
    x=Dense(1024, activation='relu')(x)
    x=Dense(128, activation='relu')(x)
    x=Dense(num_classes, activation='softmax')(x)
    model=Model(model.input,x)
    model.summary()
    return model


