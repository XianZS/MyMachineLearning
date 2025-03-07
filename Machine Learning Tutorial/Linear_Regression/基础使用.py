# -*- coding: UTF-8 -*-
"""
    @Project : MyMachineLearning 
    @File    : 基础使用.py
    @IDE     : PyCharm 
    @Author  : XianZS
    @Date    : 2025/3/7 08:37 
    @NowThing: 
"""
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

""" 数据准备 """
x_data = np.random.randint(1, 100, 30)
y_data = np.random.randint(1, 100, 30)

""" 建立模型 """
model = LinearRegression()

""" 训练模型 """
model.fit(x_data.reshape(-1, 1), y_data.reshape(-1, 1))

""" 测试模型 """
x_test = np.random.randint(1, 100, 20)
y_test = model.predict(x_test.reshape(-1, 1))
print(f"x_test: {x_test}, y_test: {y_test}")
""" 可视化 """
plt.scatter(x_data, y_data)
plt.plot(x_test, y_test, color='red')
plt.show()
