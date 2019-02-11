import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#MNIST数据相关的常数
INPUT_NODE = 784
OUTPUT_NODE =  10

#配置神经网络的参数
LATER1_NODE = 500              #隐藏层节点数
BATCH_SIZE = 100               #每一次训练集大小
LEARNING_RATE_BASE = 0.8       #基础学习率
LEARNING_RATE_DECAY = 0.99     #学习率衰减率
REGULARIZATION_RATE = 0.0001   #正则化项系数
TRAINING_STEPS = 30000         #训练次数
MOVING_AVERAGE_DECAY = 0.99    #滑动平均衰减率


#使用滑动平均模型和不使用滑动平均模型的输出结果获取
def inference(input_tensor, weights1, biases1, weights2, biases2, avg_class = None):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1)
        )
        return tf.matmul(
            layer1,avg_class.average(weights2)+avg_class.average(biases2)
         )

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x_input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y_input')

    #初始化参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE,LATER1_NODE],stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1,shape = [LATER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LATER1_NODE,OUTPUT_NODE],stddev= 0.1)
     )
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

    #不使用滑动平均模型的输出结果y
    y = inference(x,weights1,biases1,weights2,biases2)
    #全局步数
    global_step = tf.Variable(0,trainable=False)
    #定义滑动平均
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #将滑动平均应用到图中所有可训练变量
    variable_average_op = variable_average.apply(tf.trainable_variables())
    #使用滑动平均模型的输出结果average_y
    average_y = inference(x,weights1,biases1,weights2,biases2,variable_average)

    #定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)   #定制正则化器 L2正则化
    regularization = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean + regularization

    #定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    #定义训练方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    train_op = tf.group(train_step,variable_average_op)
    #计算准确率
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #创建会话
    with tf.Session() as sess:
        #执行变量初始化器
        tf.global_variables_initializer().run()
        #训练集和测试集的数据和标签的字典
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed) #定义正确率的时候用到了y_，所以要提供feed_dict
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " %(i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict = {x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict = test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g "%(TRAINING_STEPS,test_acc))


def main(argv = None):
    mnist = input_data.read_data_sets("E:\\text\\py\\DeepLearning\\MNIST_data",one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
