import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 只有1.X版本支持session操作，而此时的tensorflow版本是2.X的，所以要做兼容操作。
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

#使用numpy生成500个随机点,即输入有500个节点
x_data=np.linspace(-1,1,500)[:,np.newaxis]
print(x_data.shape)  # 500行1列
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+noise
 
#定义3个placeholder存放输入数据，shape是未知行*1列
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
z=tf.placeholder(tf.float32,[None,1])
#定义神经网络中间层，中间节点20个
Weights_L1=tf.Variable(tf.random_normal([1,20]))
print(Weights_L1.shape)
biases_L1=tf.Variable(tf.zeros([1,20]))    #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
print(Wx_plus_b_L1.shape)
L1=tf.nn.tanh(Wx_plus_b_L1)   #加入激活函数

#定义神经网络输出层，输出节点2个
Weights_L2=tf.Variable(tf.random_normal([20,2]))
biases_L2=tf.Variable(tf.zeros([1,2]))  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)   #加入激活函数

#定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）,0.1为学习率
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
 
#   获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})

#画图
plt.figure()
plt.scatter(x_data,y_data)   #散点是真实值
plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
plt.show()
