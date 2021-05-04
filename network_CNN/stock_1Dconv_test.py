# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

look_back = 10

data = np.genfromtxt('../dataset/stock_2330.csv', delimiter=',', dtype=None)
prices = np.array([price for date, price in data]).astype(np.float64)

# 把資料最後 5 + 1 筆當作 testX(前 5 天) 跟 testY(預測第 6 天)
test = prices[-1 * (look_back + 1):]
testX, testY = np.array([test[0:look_back]]), [test[look_back]]
testX = np.expand_dims(testX, axis=2)

model = load_model('stock_1DCNN_modeltest1.h5')
price = model.predict(testX)

print("Predict Price:", price[0][0])
print("Actual Price:", testY[0])
