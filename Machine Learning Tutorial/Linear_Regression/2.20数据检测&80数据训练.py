# -*- coding: UTF-8 -*-
"""
    @Project : MyMachineLearning 
    @File    : 2.20数据检测&80数据训练.py
    @IDE     : PyCharm 
    @Author  : XianZS
    @Date    : 2025/3/7 09:23 
    @NowThing: 80%数据训练 20%数据检测
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

""" 数据准备阶段 """
# x_train：训练数据的特征矩阵，通常是一个二维数组，每一行代表一个样本，每一列代表一个特征。
# y_train：训练数据的目标值向量，是一个一维数组，与 x_train 中的样本一一对应。
x_data = np.random.randint(1, 100, 100).reshape(-1, 1)
y_data = np.random.randint(1, 100, 100).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

""" 模型创建 """
model = LinearRegression()
model.fit(x_train, y_train)

""" 模型评估 """
score = model.score(x_test, y_test)
print(f"模型的评估结果为：{score}")

""" 模型预测 """
y_predict = model.predict(x_test)
print(f"预测结果为：{y_predict}")

""" 可视化 """
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_predict, color='blue')
plt.show()
