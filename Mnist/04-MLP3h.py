#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/7/6
"""
from time import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as mnist


# ==========================================================
#
# 環境
#
# ==========================================================




# ==========================================================
#
# Helper
#
# ==========================================================

def layer(output_dim, input_dim, inputs, activation=None):
    w = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.zeros([1, output_dim]))
    xwb = tf.matmul(inputs, w) + b
    if activation is None:
        outputs = xwb
    else:
        outputs = activation(xwb)
    return outputs


# ==========================================================
#
# 資料預處理
#
# ==========================================================
print('Loading data...')

data = mnist.read_data_sets('MNIST_data/', one_hot=True)

X_train = data.train.images
y_train = data.train.labels

X_val = data.validation.images
y_val = data.validation.labels

X_test = data.test.images
y_test = data.test.labels

# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')

# 建立輸入層
X = tf.placeholder(tf.float32, [None, 28*28])

# 建立隱藏層
h1 = layer(1024*1, 28*28, X, tf.nn.relu)

h2 = layer(1024, 1024*1, h1, tf.nn.relu)

# h3 = layer(1024*1, 1024*1, h2, tf.nn.relu)

# 建立輸出層
y = layer(10, 1024, h2)


# ==========================================================
#
# 定義訓練方式
#
# ==========================================================


# 建立訓練資料label真實值 placeholder
y_label = tf.placeholder(tf.float32, [None, 10])
# 定義Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y))

# 選擇optimizer
train = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

# ==========================================================
#
# 定義模型準確率
#
# ==========================================================


# 計算每一筆資料是否正確預測
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))

# 將預測結果加總平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ==========================================================
#
# 開始訓練
#
# ==========================================================
print('Training Model...')

trainEpochs = 15
batchSize = 100
totalBatch = int(len(X_train) / batchSize)
epoch_list = []
loss_list = []
accuracy_list = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

startTime = time()

for epoch in range(trainEpochs):
    for i in range(totalBatch):
        batch_X, batch_y = data.train.next_batch(batchSize)
        sess.run(train, feed_dict={X: batch_X, y_label: batch_y})

    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={X: X_val, y_label: y_val})
    epoch_list.append(epoch)
    accuracy_list.append(acc)
    loss_list.append(loss)
    epoch_time = time() - startTime
    print('Epoch ', '%02d' % (epoch + 1), ' Loss: ', '{:.9f}'.format(loss), ' Acc: ', '{:.5f}'.format(acc), ' time: ', epoch_time)

duration = time() - startTime
print("Train Finished takes:", duration)

# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
print('Evaluate Model...')

# ==========================================================
#
# 進行預測
#
# ==========================================================
print('Predict...')
