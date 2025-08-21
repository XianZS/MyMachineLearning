# -*- coding: UTF-8 -*-
"""
# Basic Information Of The File
    @Project : MyMachineLearning 
    @File    : Linear_Regression.py
    @Author  : XianZS
# Meaning
## Procedure
1. 导入数据
2. 建立模型
3. 计算损失函数
4. 梯度下降法
5. 利用梯度更新参数
6. 设置训练轮次
7. 使用更新之后的参数进行推理预测
## Problem
给出学习时间为1小时的成绩，以及学习时间为2小时的成绩，以及学习时间为3小时的成绩，
需要预测学习时间为4小时、5小时、n小时的成绩是多少？
## Author
XianZS
"""

"""
Ⅰ. 定义数据集
$
y=wx+b
$
"""
# 定义数据特征 x轴
x_data = [1, 2, 3]
# 定义数据标签 y轴
y_data = [2, 4, 6]
# 初始化参数w
w = 4


# 初始化线性回归的模型
def forward_line_mode(x):
    return w * x


# 定义损失函数
def cost(xs, ys):
    """
        损失函数计算方式
        通过累加计算结果和真实值之间的差值的平方，来计算损失，损失值越接近0，则表明计算方式越好
        损失函数 = 累加 (y_pred - y)^2
    """
    cost_value = 0
    # 损失函数初始值
    for x, y in zip(xs, ys):
        # 计算当前x的预测值 y_pred
        y_pred = forward_line_mode(x)
        # 计算损失差，也就是当前x对应的真实值y与预测值y_pred的差值
        cost_value += (y - y_pred) ** 2
    # 返回损失平均值
    return cost_value / len(xs)


# 定义梯度计算的函数公式
def gradient(xs, ys):
    """
        梯度计算方式
        梯度其实就是通过对x偏导值的计算，来迭代当前函数的k，也就是斜率w
        梯度 = 累加 (2 * (x * w - y) * x) / 数据量
    """
    grad = 0
    for x, y in zip(xs, ys):
        # （根据偏导数）计算当前的梯度
        grad += 2 * (x * w - y) * x
    # 返回梯度平均值
    return grad / len(xs)


for epoch in range(100):
    # 计算误差损失，得到平均损失
    cost_val = cost(x_data, y_data)
    # 计算梯度
    grad_val = gradient(x_data, y_data)
    # 更新梯度
    w = w - 0.01 * grad_val
    # 打印
    print("训练轮次epoch:", epoch, "w:", w, "cost:", cost_val)

print("100轮训练之后，得到的w的数值为:{}\n计算学习时间为4h时，成绩为{}".format(w, forward_line_mode(4)))
