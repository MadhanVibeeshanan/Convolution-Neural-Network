# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:51:16 2018

@author: madha
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple
np.random.seed(101)
%matplotlib inline

Dataset = namedtuple('Dataset', ['x', 'y'])

def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis=0).astype(np.float32)

dir = ['A','B','C','D','E']
subdir = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

n_classes = 24
resized_image = (32,32)

r, c = 5, 24;
Matrix = [[0 for x in range(c)] for y in range(r)] 

def load_images(folder, n_lables, resize_to):
    images = []
    labels = [] 
    for i in np.arange(5):
        print("Directory :", dir[i])
        for j in np.arange(24):
            print("Subdirectory ",subdir[j])
            root_path = os.path.join(folder, dir[i], subdir[j])
            img_name = root_path.replace('\\','/')
            count = 0
            for img_name in glob.glob(img_name+'/*.png'):
                img = plt.imread(img_name).astype(np.float32)
                #print(len(img.shape))
                if(len(img.shape) == 3):                
                    img = rgb2lab(img/255.0)[:,:,0]
                if resize_to:
                    img = resize(img, resize_to, mode='reflect')    
                label = np.zeros((n_lables, ), dtype = np.float32)
                label[j] = 1.0
            
                images.append(img.astype(np.float32))
                labels.append(label)
                count = count + 1
            Matrix[i][j] = count
    return Dataset(x = to_tf_format(images).astype(np.float32), y = np.matrix(labels).astype(np.float32))

root_folder = 'C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 5/dataset5'            
dataset = load_images(root_folder, n_classes, resized_image)

print(dataset.x.shape)
print(dataset.y.shape)

plt.imshow(dataset.x[1800, :, :, :].reshape(resized_image)) #sample
print(dataset.y[600, :])

from sklearn.model_selection import train_test_split
idx_train, idx_test = train_test_split(range(dataset.x.shape[0]), test_size=0.2, random_state=101)
X_train = dataset.x[idx_train, :, :, :]
X_test = dataset.x[idx_test, :, :, :]
y_train = dataset.y[idx_train, :]
y_test = dataset.y[idx_test, :]
         
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


def minibatcher(x, y, batch_size, shuffle):
  assert x.shape[0] == y.shape[0]
  n_samples = x.shape[0]
  
  if shuffle:
    idx = np.random.permutation(n_samples)
  else:
    idx = list(range(n_samples))
  print("IDX is", idx)
  for k in range(int(np.ceil(n_samples/batch_size))):
    from_idx = k*batch_size
    to_idx = (k+1)*batch_size
    yield x[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]
    
for mb in minibatcher(X_train, y_train, 10000, True):
  print(mb[0].shape, mb[1].shape)
  
import tensorflow as tf

def fc_no_activation_layer(in_tensors, n_units):
  w = tf.get_variable('fc_W', [in_tensors.get_shape()[1], n_units],tf.float32, tf.contrib.layers.xavier_initializer())
  b = tf.get_variable('fc_B', [n_units, ], tf.float32, tf.constant_initializer(0.0))
  return tf.matmul(in_tensors, w) + b

def fc_layer(in_tensors, n_units):
  return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))

def maxpool_layer(in_tensors, sampling):
  return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1], [1, sampling, sampling, 1], 'SAME')
  
def conv_layer(in_tensors, kernel_size, n_units):
  w = tf.get_variable('conv_W', 
    [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
    tf.float32,
    tf.contrib.layers.xavier_initializer())
  b = tf.get_variable('conv_B',
    [n_units, ],
    tf.float32,
    tf.constant_initializer(0.0))
  return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)

def dropout(in_tensors, keep_proba, is_training):
  return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)

  
def model(in_tensors, is_training):
  
  # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 20% drouput
  with tf.variable_scope('l1'):
    l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
    l1_out = dropout(l1, 0.8, is_training)
  
  # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% drouput
  with tf.variable_scope('l2'):
    l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
    l2_out = dropout(l2, 0.8, is_training)
    
  with tf.variable_scope('flatten'):
    l2_out_flat = tf.layers.flatten(l2_out)
  
  # Fully collected layer, 1024 neurons, 40% dropout 
  with tf.variable_scope('l3'):
    l3 = fc_layer(l2_out_flat, 1024)
    l3_out = dropout(l3, 0.6, is_training)
  
  # Output
  with tf.variable_scope('out'):
    out_tensors = fc_no_activation_layer(l3_out, n_classes)
  
  return out_tensors

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import pandas as pd
report = pd.DataFrame

def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size):

  in_X_tensors_batch = tf.placeholder(tf.float32, shape = (None, resized_image[0], resized_image[1], 1))
  in_y_tensors_batch = tf.placeholder(tf.float32, shape = (None, n_classes))
  is_training = tf.placeholder(tf.bool)

  logits = model(in_X_tensors_batch, is_training)
  out_y_pred = tf.nn.softmax(logits)
  loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=in_y_tensors_batch)
  loss = tf.reduce_mean(loss_score)
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    total_score = []
    for epoch in range(max_epochs):
      print("Epoch=", epoch)
      tf_score = []
      
      for mb in minibatcher(X_train, y_train, batch_size, shuffle = True):
        tf_output = session.run([optimizer, loss], 
                                feed_dict = {in_X_tensors_batch : mb[0], 
                                             in_y_tensors_batch : mb[1],
                                             is_training : True})
      
        tf_score.append(tf_output[1])
        total_score.append(tf_output[1])
      print(" train_loss_score=", np.mean(tf_score))
    t = np.arange(len(total_score))
    plt.plot(t, total_score) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.grid(True)
    plt.show()    
    # after the training is done, time to test it on the test set
    print("TEST SET PERFORMANCE")
    y_test_pred, test_loss  = session.run([out_y_pred, loss], 
                                          feed_dict = {in_X_tensors_batch : X_test, 
                                                       in_y_tensors_batch : y_test,
                                                       is_training : False})
    
    print(" test_loss_score=", test_loss)
    y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
    y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
    print(classification_report(y_test_true_classified, y_test_pred_classified))
    
    
    cm = confusion_matrix(y_test_true_classified, y_test_pred_classified)
    Accuracy = accuracy_score(y_test_true_classified, y_test_pred_classified)   
    precision, recall, fscore, support = score(y_test_true_classified, y_test_pred_classified)    
    
    print("Overall Accuracy :", np.mean(Accuracy)*100)
    print('Overall precision: ',np.mean(precision)*100)
    print('Overall recall: ',np.mean(recall)*100)
    print('Overall fscore: ',np.mean(fscore)*100)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # And the log2 version, to enphasize the misclassifications
    plt.imshow(np.log2(cm + 1), interpolation='nearest', cmap=plt.get_cmap("tab20"))
    plt.colorbar()
    plt.tight_layout()
    plt.show()


tf.reset_default_graph()
train_model(X_train, y_train, X_test, y_test, 0.001, 10, 256)

