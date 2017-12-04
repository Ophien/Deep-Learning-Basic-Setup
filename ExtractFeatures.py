
#some basic imports and setups
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

#To open the Alexnet model
from alexnet import AlexNet

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

#Set your dataset address
database_dir = '/ADDRESS/TO/DATASET/FOLDER/CLOTHS_CROPPED/'
dirs = os.listdir(database_dir)


data_x = []
data_y = []

with open(database_dir+'training.txt', 'r') as image_list:
    i=0
    for line in image_list:
        nimg = cv2.imread(database_dir+line.split(" ")[0])
        data_x.append(nimg)
        data_y.append(line.split(" ")[-1])

class_names = ['other', 'animal', 'cartoon', 'chevron', 'floral', 'geometry', 'houndstooth', 'ikat', 'numb', 'plain', 
              'polka dot', 'scales', 'skull', 'squares', 'stars', 'stripes', 'tribal']

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with 17 outputs and loading weights except for the last softmax layer
model = AlexNet(x, keep_prob, 17, ['fc8'])

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax 
softmax = tf.nn.softmax(score)

#Define the layer to extract features
#[model.pool1, model.pool5, model.fc7]]

layer = model.pool5
layername = 'conv5'

with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load the pretrained weights into the model
    model.load_initial_weights(sess)
    
    j=0
    
    datas=[[],[]]
    
    # Loop over all images
    for i, image in enumerate(data_x):
        
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227,227))
        
        # Subtract the ImageNet mean
        img -= imagenet_mean
        
        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))
        
        # Run the session
        features = sess.run([layer], feed_dict={x: img, keep_prob: 1})
        
        if(i % 100 == 0):
            print (i)
        
        if(i % 5000 == 0 and i != 0):
            with open(layername + str(i/5000)+".test.pickle", "wb") as output_file:
                pickle.dump(datas, output_file)
            datas=[[],[]]
        
        
        datas[0].append(features[0][0])
        datas[1].append(np.int32(data_y[i]))
                
    with open(layername + str(i) + ".test.pickle", "wb") as output_file:
        pickle.dump(datas, output_file)



