#!/usr/bin/env python
# coding: utf-8

# ## Import essential library

# In[1]:


from PIL import Image, ImageDraw
import os
import glob
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf


# ## Get all images 

# In[2]:


images_paths = glob.glob(r'./Dataset -teeth/clean/train/images/*.JPG')


# In[3]:


images_paths


# ## Every image file in the dataset is 480 X 480, so there is no need to resize the images

# In[3]:


images = []
for imagefile in images_paths:
    image = Image.open(imagefile)
    image = np.asarray(image)/255.0
    images.append(image)


# ## Get the bounding box location, prepare the targets

# In[4]:


bboxes = []
classes_raw = []
annotations_paths = glob.glob( r'./Dataset -teeth/clean/train/images/*.xml')
for xmlfile in annotations_paths:
    x = xmltodict.parse( open( xmlfile , 'rb' ) )
    bndbox = x[ 'annotation' ][ 'object' ][ 'bndbox' ]
    bndbox = np.array([ int(bndbox[ 'xmin' ]) , int(bndbox[ 'ymin' ]) , int(bndbox[ 'xmax' ]) , int(bndbox[ 'ymax' ]) ])
    bndbox2 = [ None ] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]
    bndbox2 = np.array( bndbox2 ) / 480
    bboxes.append( bndbox2 )
    classes_raw.append( x[ 'annotation' ][ 'object' ][ 'name' ] )


# ## Create training and testing data

# In[5]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# In[6]:


boxes = np.array( bboxes ) 
encoder = LabelBinarizer()
classes_onehot = encoder.fit_transform( classes_raw )


# In[7]:


Y = np.concatenate( [ boxes , classes_onehot ] , axis=1 )
X = np.array( images )


# In[9]:


Y


# In[8]:


print(Y.shape)
print(X.shape)


# In[9]:


type(X[0][0][0][0])


# In[10]:


type(Y[0][0])


# In[11]:


x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.1 )


# ## Define the loss function and metrics
# 
# ### Loss function:
# Combine the Mean Square Error and Intersection over Union
# 
# $L(x,x') = MSE(x,x') + (1 - IOU(x, x'))$
# 
# $IOU(x,x') = {x \bigcap x' \over x \bigcup x'}$
# 
# ### Evaluation Metrics
# $IOU(x,x') = {x \bigcap x' \over x \bigcup x'}$

# In[12]:


import tensorflow.keras.backend as K


# ### Define the IOU function

# ### Define the loss function and evaluation metrics

# In[3]:


def calculate_iou( target_boxes , pred_boxes ):
    xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = K.maximum( 0.0 , xB - xA ) * K.maximum( 0.0 , yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    return iou

def custom_loss( y_true , y_pred ):
    mse = tf.losses.mean_squared_error( y_true , y_pred ) 
    iou = calculate_iou( y_true , y_pred ) 
    return mse + ( 1 - iou )

def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred )


# ## Create the model
# 
# [Description about the model]

# In[14]:


input_shape = ( 480 , 480 , 3 )
dropout_rate = 0.5
alpha = 0.2
num_classes = 1
pred_vector_length = 4 + num_classes


# In[15]:


model_layers = [       
	keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1 ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.MaxPooling2D( pool_size=( 2 , 2 ) ),

    keras.layers.Flatten() , 

    keras.layers.Dense( 1240 ) , 
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 640 ) , 
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 480 ) , 
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 120 ) , 
    keras.layers.LeakyReLU( alpha=alpha ) ,
    keras.layers.Dense( 62 ) , 
    keras.layers.LeakyReLU( alpha=alpha ) ,

    keras.layers.Dense( pred_vector_length ),
    keras.layers.LeakyReLU( alpha=alpha ) ,
]

model = keras.Sequential( model_layers )
model.compile(
	optimizer=keras.optimizers.Adam( lr=0.0001 ),
	loss=custom_loss,
    metrics=[ iou_metric ]
)


# ## Train the model

# In[16]:


model.fit( 
    x_train ,
    y_train , 
    validation_data=( x_test , y_test ),
    epochs=100 ,
    batch_size=3 
)


# ## Save model

# In[23]:


model.save( 'model.h5')


# ## Define a function to draw bounding box and output the data in a dir

# In[19]:


def predict_n_drawBnd(model, imagedata, outputfilename):
    os.mkdir(outputfilename)
    boxes = model.predict( imagedata )
    for i in range( boxes.shape[0] ):
        b = boxes[ i , 0 : 4 ] * 480 
        img = imagedata[i] * 255
        source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
        draw = ImageDraw.Draw( source_img )
        draw.rectangle( b , outline="black" )
        filename = outputfilename +'/image_{}.png'
        source_img.save( outputfilename +'/image_{}.png'.format( i + 1 ) , 'png' )


# ## Predict on validation data

# In[20]:


predict_n_drawBnd(model, x_test, 'inference_images')


# ## Let the model to meet the test data

# In[14]:


valid_images_paths = glob.glob(r'./Dataset -teeth/clean/valid/images/*.JPG')


# In[15]:


valid_images = []
for imagefile in valid_images_paths:
    image = Image.open(imagefile)
    image = np.asarray(image)/255.0
    valid_images.append(image)


# In[16]:


x_valid = np.array( valid_images )


# In[17]:


x_valid.shape


# In[20]:


predict_n_drawBnd(model, x_valid, 'inference_valid_images')


# ## Load model and compile

# In[1]:


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model


# In[9]:


model = load_model('model.h5', compile = False)


# In[10]:


model.compile(
	optimizer=keras.optimizers.Adam( lr=0.0001 ),
	loss=custom_loss,
    metrics=[ iou_metric ]
)

