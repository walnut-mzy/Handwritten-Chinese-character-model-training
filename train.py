from model import resnet101v2_model
from utlis import train,test
import setting
import tensorflow as tf
tf.executing_eagerly()
model=resnet101v2_model(3754)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
train_dataset=train()
test_dataset=test()
if __name__ == '__main__':

    for i in range(setting.EPOCHS):
        print("epoch:{}  ".format(i))
        model.fit(train_dataset,batch_size=setting.BATCH_SIZE, validation_data=test_dataset)

        #if i%10==0:
