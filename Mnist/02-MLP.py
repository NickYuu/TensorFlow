#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/6/26
"""
from time import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


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


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)

        ax.imshow(np.reshape(images[idx], (28, 28)),
                  cmap='binary')

        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# ==========================================================
#
# 資料預處理
#
# ==========================================================
print('Loading data...')

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

X_train = mnist.train.images
y_train = mnist.test.labels

X_val = mnist.validation.images
y_val = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')

# 建立輸入層
X = tf.placeholder(tf.float32, [None, 28 * 28])

# 建立隱藏層
h1 = layer(1024, 28 * 28, X, tf.nn.relu)

h2 = layer(512, 1024, h1, tf.nn.relu)

h3 = layer(256, 512, h2, tf.nn.relu)

# 建立輸出層
y = layer(10, 256, h3)

# ==========================================================
#
# 定義訓練方式
#
# ==========================================================

# 建立訓練資料label真實值 placeholder
y_label = tf.placeholder(tf.float32, [None, 10])

# 定義Loss function
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y))

# 選擇optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

# ==========================================================
#
# 定義模型準確率
#
# ==========================================================


# 計算每一筆資料是否正確預測
correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y, 1))

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
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={X: batch_x, y_label: batch_y})

    loss, acc = sess.run([loss_function, accuracy], feed_dict={X: X_val, y_label: y_val})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch:", '%02d' % (epoch + 1), "Loss=", "{:.9f}".format(loss), " Accuracy=", "{:.05f}".format(acc))

duration = time() - startTime
print("Train Finished takes:", duration)

# ==========================================================
#
# 評估模型準確率
#
# ==========================================================
print('Evaluate Model...')
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss'])
plt.show()

plt.plot(epoch_list, accuracy_list)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['acc'])
plt.show()

print('Accuracy : ', sess.run(accuracy, feed_dict={X: X_test, y_label: y_test}))

# ==========================================================
#
# 進行預測
#
# ==========================================================
print('Predict...')
prediction = sess.run(tf.argmax(y, axis=1), feed_dict={X: X_test})
label = sess.run(tf.argmax(y_label, axis=1), feed_dict={y_label: y_test})

df = pd.DataFrame({'label': label, 'predict': prediction})
index = df[df.label != df.predict].index

for i in index[:10]:
    print('真實值為: ', label[i], '預測值為: ', prediction[i])


plot_images_labels_prediction(X_test[index], label[index], prediction[index], idx=0)
print()
matrix = pd.crosstab(df.label, df.predict)
print(matrix)
