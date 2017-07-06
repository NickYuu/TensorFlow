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

def layer(input_dim, output_dim, inputs, activation=None):
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

# ==========================================================
#
# 建立模型
#
# ==========================================================
print('Build model...')

# 建立輸入層

# 建立隱藏層

# 建立輸出層



# ==========================================================
#
# 定義訓練方式
#
# ==========================================================


# 建立訓練資料label真實值 placeholder

# 定義Loss function

# 選擇optimizer

# ==========================================================
#
# 定義模型準確率
#
# ==========================================================


# 計算每一筆資料是否正確預測

# 將預測結果加總平均



# ==========================================================
#
# 開始訓練
#
# ==========================================================
print('Training Model...')

trainEpochs = 0
batchSize = 0
totalBatch = 0
epoch_list = []
loss_list = []
accuracy_list = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

startTime = time()

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
