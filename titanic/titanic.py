#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm
@since:python 3.6.0 on 2017/6/30
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================================
#
# 環境
#
# ==========================================================

np.random.seed(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==========================================================
#
# Helper
#
# ==========================================================


def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


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

# ==========================================================
#
# 訓練模型
#
# ==========================================================
print('Training Model...')

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

