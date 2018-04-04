import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer


def weight_variable(name, shape):
    """Create a weight variable with Normal distribution """
    initer = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('W_' + name, dtype=tf.float32, shape=shape, initializer=initer)


def biases_variable(name, shape):
    """Create a biases variable with zero distribution"""
    initer = tf.constant_initializer(0.0)
    return tf.get_variable('b_' + name, dtype=tf.float32, shape=shape, initializer=initer)


def batch_normalization(name, input_tensor, out_dim, varience_epsilon):
    """Create a bn layer"""
    mean, var = tf.nn.moments(input_tensor, axes=list(range(input_tensor.get_shape().ndims-1)))
    offset = tf.Variable(tf.zeros([out_dim]))
    scale = tf.Variable(tf.ones([out_dim]))
    return tf.nn.batch_normalization(input_tensor, mean=mean, variance=var, offset=offset, scale=scale,
                                     variance_epsilon=varience_epsilon, name=name)


def new_conv_layer(input_tensor, layer_name, stride_time, stride, num_inChannel, filter_size, num_filters, batch_norm, use_relu):
    """Create a convolution layer"""
    weight_shape = [filter_size, filter_size, filter_size, num_inChannel, num_filters]
    with tf.variable_scope(layer_name) as scope:
        weight = weight_variable(layer_name, shape=weight_shape)
        biases = biases_variable(layer_name, shape=[num_filters])
        conv = tf.nn.conv3d(input=input_tensor, filter=weight, strides=[1, stride_time, stride, stride, 1], padding='SAME')
        if batch_norm:
            batch_normalization(input_tensor=conv, out_dim=num_filters,
                                varience_epsilon=0.0001, name=scope.name+'bn')
        biases_added = tf.nn.bias_add(conv, biases)
        if use_relu:
            biases_added = tf.nn.relu(biases_added)
    return biases_added


def flatten_layer(input_tensor):
    """Flatten outputs of convolution layers and prepare to feed in fc layers"""
    with tf.variable_scope('Flatten_layer') as scope:
        layer_shape = input_tensor.get_shape()
        num_features = layer_shape[1:5].num_elements()
        layer_flat = tf.reshape(input_tensor, [-1, num_features], name=scope.name)
        return layer_flat, num_features


def fc_layer(input_tensor, out_dim, name, batch_norm=False, use_reg=True, use_relu=True):
    """Create a fc layer"""
    in_dim = input_tensor.get_shape()[1]
    with tf.variable_scope(name) as scope:
        weights = weight_variable(name, shape=[in_dim, out_dim])
        biases = biases_variable(name, shape=[out_dim])
        if use_reg:
            regularation_rate = 0.0001
            regularizer = l2_regularizer(regularation_rate)
            tf.add_to_collection('losses', regularizer(weights))
        layer = tf.matmul(input_tensor, weights)
        if batch_norm:
            batch_normalization(input_tensor=input_tensor, out_dim=out_dim,
                                varience_epsilon=0.0001, name=scope.name+'bn')
        biases_added = layer + biases
        if use_relu:
            biases_added = tf.nn.relu(layer)
    return biases_added


def max_pool(input_tensor, ksize, stride, name):
    """Create a max_pool layer"""
    return tf.nn.max_pool3d(input_tensor, ksize=[1, ksize, ksize, ksize, 1],
                            strides=[1, stride, stride, stride, 1], padding='SAME', name=name)


def avg_pool(input_tensor, ksize, stride, name):
    """Create a avg_pool layer"""
    return tf.nn.avg_pool3d(input_tensor, ksize=[1, ksize, ksize, ksize, 1],
                            strides=[1, stride, stride, stride, 1], padding='SAME', name=name)


def dropout(input_tensor, keep_prob):
    """Create a dropout_layer"""
    return tf.nn.dropout(input_tensor, keep_prob=keep_prob)
