#这是一个很简单的单层神经网络
#使用了前向神经网络算法和反向传播算法
#激活函数为sigmoid
#将交叉熵作为损失函数并用梯度下降优化
#数据是随机生成的
import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1))
x = tf.placeholder(tf.float32,shape = (None,2),name = 'x-input')
y_ = tf.placeholder(tf.float32,shape = (None,1),name = 'y-input')
#sigmoid激活函数
a = tf.sigmoid(tf.matmul(x,w1))
y = tf.sigmoid(tf.matmul(a,w2))

#损失函数
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y,0.0000000001,1.0))
    +(1-y)*tf.log(tf.clip_by_value(1-y,0.0000000001,1.0)))

#梯度下降&反向传播
train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

#随机生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

#创建会话运行TensorFlow
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 10000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i%1000 == 0:
            total_cross_entropy = sess.run(
                  cross_entropy,feed_dict = {x:X,y_:Y})
            print("After %d training step(s), cross entropy on all data is %g"%(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))

