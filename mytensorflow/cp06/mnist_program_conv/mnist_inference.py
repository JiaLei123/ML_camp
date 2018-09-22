import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable(name="weight", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biase = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1,1,1,1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biase))
    with tf.variable_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    with tf.variable_scope("layer3_conv2"):
        conv2_weight = tf.get_variable(name="weight", shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biase = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biase))
    with tf.variable_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pool_shape = pool2.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshape = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope("layer5-fc1"):
        fc1_weight = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weight))
        fc1_bizses = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weight) + fc1_bizses)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weight = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weight))
        fc2_bizses = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1, fc2_weight) + fc2_bizses

    return logit