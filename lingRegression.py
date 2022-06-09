from scipy.linalg import hilbert
x = hilbert(10)

import numpy as np
np.linalg.det(np.matrix(x).T * np.matrix(x))

# 上方  10×10  的希尔伯特矩阵
'''
皮尔逊相关系数（Pearson Correlation Coefficient）通常用于度量两个变量 
 𝑋  和  𝑌  之间的线性相关程度，其值介于 -1 与 1 之间。
 其中，数值越趋近于 1 表示正相关程度越高，趋近于 0 表示线性相关度越低，趋近于 -1 则表示负相关程度越高。
'''

import pandas as pd

pd.DataFrame(x, columns=['x%d'%i for i in range(1,11)]).corr()

from scipy.optimize import leastsq

x = hilbert(10) # 生成 10x10 的希尔伯特矩阵
np.random.seed(10) # 随机数种子能保证每次生成的随机数一致
w = np.random.randint(2,10,10) # 随机生成 w 系数
y_temp = np.matrix(x) * np.matrix(w).T # 计算 y 值
y = np.array(y_temp.T)[0] #将 y 值转换成 1 维行向量

print("实际参数 w: ", w)
print("实际函数值 y: ", y)

func=lambda p,x: np.dot(x, p) # 函数公式
err_func = lambda p, x, y: func(p, x)-y # 残差函数
p_init=np.random.randint(1,2,10) # 全部参数初始化为 1

parameters = leastsq(err_func, p_init, args=(x, y)) # 最小二乘法求解
print("拟合参数 w: ",parameters[0])

from sklearn.linear_model import Ridge

ridge_model = Ridge(fit_intercept=False) # 参数代表不增加截距项
ridge_model.fit(x, y)

print(ridge_model.coef_ )# 打印模型参数

"""不同 alpha 参数拟合
"""
alphas = np.linspace(-3,2,20)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)

"""绘制不同 alpha 参数结果
"""
from matplotlib import pyplot as plt
# %matplotlib inline

plt.plot(alphas, coefs) # 绘制不同 alpha 参数下的 w 拟合值
plt.scatter(np.linspace(0,0,10), parameters[0]) # 普通最小二乘法拟合的 w 值放入图中
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Ridge Regression')

"""使用 LASSO 回归拟合并绘图
"""
from sklearn.linear_model import Lasso

alphas = np.linspace(-2, 2, 10)
lasso_coefs = []

for a in alphas:
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(x, y)
    lasso_coefs.append(lasso.coef_)

plt.plot(alphas, lasso_coefs)  # 绘制不同 alpha 参数下的 w 拟合值
plt.scatter(np.linspace(0, 0, 10), parameters[0])  # 普通最小二乘法拟合的 w 值放入图中
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Lasso Regression')


