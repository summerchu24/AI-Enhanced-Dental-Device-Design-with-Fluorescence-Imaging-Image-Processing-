#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageDraw
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


source_path = '/content/drive/MyDrive/Colab Notebooks/augment/*'


# In[4]:


images_paths = glob.glob(source_path)


# In[5]:


class_name = ['Gingivits', 'Cold_Sores', 'Canker_Sores', 'Periodontitis', 'Receding_Gum', 'abfraction', 'Thrush', 'Gingival_Cyst']


# In[6]:


images_paths


# In[7]:


eight_class = []
class_volume = []


# In[8]:


for i in images_paths:
    images = glob.glob( i +'/*.jpeg')
    class_volume.append(len(images))
    eight_class.append(images)


# In[9]:


len(eight_class)


# In[10]:


images = []
labels = []


# In[11]:


class_volume


# In[12]:


import matplotlib.ticker as ticker


# In[13]:


plt.figure()
x = np.arange(8)
y = class_volume
plt.bar(x, y, width=0.5, tick_label=class_name)
plt.grid(linestyle='-.',alpha=0.5)
plt.xticks(x, class_name, rotation = 60)
plt.show()


# In[14]:


for classIndex in range(len(eight_class)):
    for imagefile in eight_class[classIndex]:
        image = Image.open(imagefile).resize((90,90))
        image = image = np.asarray(image)/255.0
        images.append(image)
        labels.append(classIndex)


# In[15]:


images = np.array(images)
labels = np.array(labels)


# In[16]:


encoder = LabelBinarizer()
labels_onehot = encoder.fit_transform( labels )


# In[17]:


images.shape,  labels_onehot.shape


# In[18]:


plt.figure()
plt.imshow(images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[27]:


x_train, x_test, y_train, y_test = train_test_split( images, labels, test_size=0.2 )


# In[20]:


plt.figure(figsize=(15,15))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.array(x_train)[i])
    plt.xlabel(np.array(class_name)[np.array(y_train)[i]])
plt.show()


# In[28]:


x_train, x_test, y_train, y_test = train_test_split( images, labels, test_size=0.2 )


# In[29]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[23]:


model = tf.keras.models.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Activation('softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_split=0.2)


# In[24]:


model.summary()


# In[25]:


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# ## Data Augmentation

# In[30]:


data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(90, 90, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[37]:


model = tf.keras.models.Sequential([
    data_augmentation,                                
    tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Activation('softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=200, validation_split=0.2)


# In[38]:


model.summary()


# In[39]:


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[40]:


# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[41]:


# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

