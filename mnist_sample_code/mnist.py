# coding: utf-8
# Parameters
LAMBDA = 0.5
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 10
train_dir = "/home/aiko/tensorflow/train"
test_dir = "/home/aiko/tensorflow/test"

# Import modules
import os
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Construct network
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
# 获取特征的维数，例如256维    
global_step = tf.Variable(0, trainable=False, name='global_step')


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
    
    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    ####################################
    loss = 0
    for i in range(128):
        fi = features[i]  # (2)
        ci = centers_batch[i]  # (2)
        fi = tf.reshape(fi, (2, 1))
        ci = tf.reshape(ci, (1, 2))
        ci_fi = tf.matmul(ci, fi)
        ci_fi = tf.reshape(ci_fi, [1])
        ci_norm = tf.sqrt(tf.reduce_sum(tf.square(ci), axis=1))
        fi_norm = tf.sqrt(tf.reduce_sum(tf.square(fi), axis=0))
        loss += 1 - ci_fi / tf.multiply(ci_norm, fi_norm)
    loss = tf.reduce_sum(loss)
    # centers_batch_norm = tf.sqrt(tf.reduce_sum(tf.square(centers_batch), axis=0))
    # features_norm = tf.sqrt(tf.reduce_sum(tf.square(features), axis=0))
    # x3_x4 = tf.reduce_sum(tf.multiply(centers_batch_norm, features_norm), axis=0)
    # cosine = x3_x4 / (centers_batch_norm * features_norm)
    # l2 normalzation
    # loss = tf.nn.l2_loss(features - centers_batch)
    # loss = tf.reduce_sum(1 - cosine)
    ####################################
    # 当前mini-batch的特征值与它们对应的中心值之间的差 delte Cj= diff
    # change this row as a new loss function

    diff = centers_batch - features

    ####################################

    # print (diff.shape)
    # diff = 1 - cosine
    ####################################

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    # tf.gather 根据索引从参数轴上收集切片
    appear_times = tf.gather(unique_count, unique_idx)
    # tf.reshape 将tensor变换为参数shape的形式
    appear_times = tf.reshape(appear_times, [-1, 1])
    # tf.cast 数据格式转化 
    diff = diff / tf.cast((1 + appear_times), tf.float32)

    diff = alpha * diff
    # 将ref中特定位置的数分别进行减法运算(original data, position from 0, update)
    #############################################
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    # theta = tf.matmul(centers_batch , centers_batch , True , False , False , False)
    # cosine_theta = theta / (centers_batch_norm * centers_batch_norm)
    # centers_update_op =features * (1 - cosine_theta)
    #############################################
    return loss, centers, centers_update_op


def inference(input_images):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')

            x = tflearn.prelu(feature)

            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')

    return x, feature


def build_network(input_images, labels, ratio=0.5):
    logits, features = inference(input_images)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            # ratio is the lambda parameter
            # total_loss = softmax_loss + ratio * center_loss
            total_loss = center_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss, accuracy, centers_update_op


logits, features, total_loss, accuracy, centers_update_op = build_network(input_images, labels, ratio=LAMBDA)

# Prepare data

mnist = input_data.read_data_sets('/tmp/mnist', reshape=False)

# Optimizer
optimizer = tf.train.AdamOptimizer(0.001)

with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
summary_op = tf.summary.merge_all()

# Session and Summary

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('/tmp/mnist_log', sess.graph)

# start training
mean_data = np.mean(mnist.train.images, axis=0)

step = sess.run(global_step)
while step <= 8000:
    batch_images, batch_labels = mnist.train.next_batch(128)
    costValue, summary_str, train_acc = sess.run(
        [train_op, summary_op, accuracy],
        feed_dict={
            input_images: batch_images - mean_data,
            labels: batch_labels,
        })
    step += 1

    writer.add_summary(summary_str, global_step=step)

    if step % 20 == 0:
        vali_image = mnist.validation.images - mean_data
        vali_acc = sess.run(
            accuracy,
            feed_dict={
                input_images: vali_image,
                labels: mnist.validation.labels
            })
        print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
               format(step, train_acc, vali_acc)))
        print("The cost is: ", costValue)

# Visualize train_data
feat = sess.run(features, feed_dict={input_images: mnist.train.images[:10000] - mean_data})

import matplotlib.pyplot as plt

labels = mnist.train.labels[:10000]

f = plt.figure(figsize=(16, 9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()

# Visualize test_data
feat = sess.run(features, feed_dict={input_images: mnist.test.images[:10000] - mean_data})

import matplotlib.pyplot as plt

labels = mnist.test.labels[:10000]

f = plt.figure(figsize=(16, 9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()

# save model
# with th.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    saver.save(sess,"/home/aiko/TensorFlow_Center_Loss/mnist_model.ckpt")

#
sess.close()
