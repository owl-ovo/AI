# 引入类库
from numpy import random, dot, exp, array

# 加载数据
# X：3个人（2女1男），0，没去看电影；1，去看电影了
# y：另外1个人（男），通过差值，寻找他跟其他人去看电影的关系
X = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])        # 4行3列
y = array([0, 1, 1, 0]).T

# 设置随机权重（y跟x里哪个人更亲近，最后权重就越高）
random.seed(1)                               # 设置任意值后每次运行文件首次的结果是一样的，后续随机
weights = 2 * random.random((3, 1)) - 1      # 3行1列，3个随机数，默认区间[0,1]运算后随机数取值范围[-1,1]

# 循环
for it in range(10):
    # 利用矩阵点乘，分配随机权重，z为’关系‘系数
    z = dot(X, weights)                      # dot 矩阵相乘，点乘
    print('Z', z)
    # 激活函数：Sigmoid函数
    output = 1/(1+exp(-z))                   # exp(-z) 自然数e的-z次幂
    print('O', output)

    # 查看计算误差，通过误差处理，排除偏离值，使权重倾向近似值
    error = y - output

    # Sigmoid函数斜率，函数的导数的每一点对应该函数在这点的斜率
    slope = output * (1-output)

    # 计算增量
    delta = error * slope

    # 更新权重
    weights = weights + dot(X.T, delta)
    print(weights)

