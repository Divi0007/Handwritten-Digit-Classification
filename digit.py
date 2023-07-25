import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# a=tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = a.load_data()
# x_train=tf.keras.utils.normalize(x_train, axis=1)
# x_test=tf.keras.utils.normalize(x_test, axis=1)
# nn=tf.keras.models.Sequential()
# nn.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# nn.add(tf.keras.layers.Dense(128,activation='relu'))
# nn.add(tf.keras.layers.Dense(128,activation='relu'))
# nn.add(tf.keras.layers.Dense(10,activation='softmax'))
# nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# nn.fit(x_train,y_train, epochs=10)
# nn.save('handwritten.model')
model=tf.keras.models.load_model('handwritten.model')
# loss, accuracy=model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)
image=0
while os.path.isfile('/Users/divyanshyadav/PycharmProjects/pythonProject6/handwritten/'+str(image)+'.png'):
    try:
        img=cv2.imread('/Users/divyanshyadav/PycharmProjects/pythonProject6/handwritten/'+str(image)+'.png',0)
        img=np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The digit is ", np.argmax(prediction))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image += 1