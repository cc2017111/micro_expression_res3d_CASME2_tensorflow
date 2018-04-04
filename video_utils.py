import tensorflow as tf
import numpy as np


def randomize(x, y):
    """Randomize the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y, img_size, num_channel, num_class):
    """Reformat the data to the format acceptable for 3d cnn"""
    dataset = x.reshape((-1, img_size, img_size, img_size, num_channel)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def accuracy_generator(label_tensor, logits_tensor):
    """Calculate the classification accuracy"""
    labels = tf.cast(label_tensor, tf.int64)
    correct_prediction = tf.equal(tf.argmax(logits_tensor, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def cross_entropy_loss(label_tensor, logits_tensor):
    """Calculate the cross_entropy loss function"""
    labels = tf.cast(label_tensor, tf.int64)
    diff = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=labels)
    loss = tf.reduce_mean(diff)
    return loss
