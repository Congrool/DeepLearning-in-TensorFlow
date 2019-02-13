import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512
BATCH_SIZE = 100               #每一次训练集大小
LEARNING_RATE_BASE = 0.8       #基础学习率
LEARNING_RATE_DECAY = 0.99     #学习率衰减率
REGULARIZATION_RATE = 0.0001   #正则化项系数
TRAINING_STEPS = 30000         #训练次数

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            'weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev= 0.1)
        )
        conv1_biases = tf.get_variable(
            'bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0)
        )

        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides = [1,1,1,1], padding = 'SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1,ksize= [1,2,2,1], strides= [1,2,2,1], padding= 'SAME'
        )
    
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weights',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer= tf.truncated_normal_initializer(stddev= 0.1)
        )
        conv2_biases = tf.get_variable(
            'bias',[CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding = 'SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('later4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME'
        )

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            'weights',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev = 0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias',[FC_SIZE],initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1,0.5)
    
    with tf.variable_scope('layer6-fc2'):
        fc2_weights= tf.get_variable(
            'weights',[FC_SIZE,NUM_LABELS],
            initializer = tf.truncated_normal_initializer(stddev = 0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias',[NUM_LABELS],
            initializer= tf.truncated_normal_initializer(0.1)
        )
        logit = tf.matmul(fc1,fc2_weights) + fc2_biases

    return logit
    
def train(mnist):
    x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name= 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name= 'y-input')


    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logit = inference(x,True,regularizer)
    global_step = tf.Variable(0,trainable=False)
    #定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    mnist.train.num_examples/BATCH_SIZE,
    LEARNING_RATE_DECAY,
    )
    #优化算法
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #计算准确率 ??? 怎么计算，怎么喂数据
    correct_prediction = tf.equal(tf.argmax(logit,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

def main(argv = None):
    mnist = input_data.read_data_sets("E:/text/py/DeepLearning/MNIST_data",one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
