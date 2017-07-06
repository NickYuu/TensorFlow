#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/7/5
"""
from time import time
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print('Loading data...')

# 創造 100個亂數 值介於0~1之間
X_train = np.random.rand(100).astype(np.float32)

# label值  這裡要預測的是 weight:2  biases:3
y_train = X_train * 2 + 3


# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')


W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.zeros([1]))

y = X_train * W + b


# 定義Loss function
loss = tf.reduce_mean(tf.square(y - y_train))

# 選擇optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# ==========================================================
#
# 定義模型準確率
#
# ==========================================================


# 計算每一筆資料是否正確預測
# correct_prediction = tf.equal(y_label, y)

# 將預測結果加總平均
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ==========================================================
#
# 開始訓練
#
# ==========================================================
print('Training Model...')

loss_list = []
accuracy_list = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.square(y - y_train)))
startTime = time()

for step in range(2001):
    sess.run(optimizer)
    if step % 40 == 0:
        print(step, sess.run(W), sess.run(b),'loss:', sess.run(loss))

duration = time() - startTime
sess.close()
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
