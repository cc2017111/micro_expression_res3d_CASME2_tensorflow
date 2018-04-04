import tensorflow as tf
import video_net_pre_function
from video_utils import accuracy_generator, cross_entropy_loss
from video_network import create_network


class Resnet_18:

    __network = None
    __train_op = None
    __loss = None
    __accuracy = None
    __prob = None

    def __init__(self, numClass, frame, imgSize, imgChannel):
        self.imgSize = imgSize
        self.imgChannel = imgChannel
        self.numClass = numClass
        self.fc_dim = 512
        self.frame = frame
        self.lmbda = 0.0001
        self.init_lr = 0.001
        self.keep_prob = 0.5
        self.x, self.y = self.create_placeholder()

    def create_placeholder(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.frame, self.imgSize, self.imgSize,
                                                       self.imgChannel), name='x_input')
            self.y = tf.placeholder(tf.float32, shape=(None,), name='y_input')

        return self.x, self.y

    def inference(self):
        if self.__network is not None:
            return self

        with tf.variable_scope('Resnet'):
            net = create_network(self.x, self.fc_dim, self.keep_prob, self.numClass)
        self.__network = net
        return self

    def pred_func(self):
        if self.__prob is not None:
            return self
        self.__prob = tf.nn.softmax(self.__network)
        return self

    def accuracy_func(self):
        if self.__accuracy is not None:
            return self
        with tf.name_scope('Accuracy'):
            self.__accuracy = accuracy_generator(self.y, self.__network)
        return self

    def loss_func(self):
        if self.__loss is not None:
            return self
        with tf.name_scope('Loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = cross_entropy_loss(self.y, self.__network)
                tf.summary.scalar('cross_entropy', cross_entropy)
            with tf.name_scope('l2_loss'):
                l2_loss = tf.reduce_mean(self.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('losses')]))
                tf.summary.scalar('l2_loss', l2_loss)
            with tf.name_scope('total'):
                self.__loss = cross_entropy + l2_loss
        return self

    def train_func(self):
        if self.__train_op is not None:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    def probs(self):
        return self.__prob

    def network(self):
        return self.__network

    def train_op(self):
        return self.__train_op

    def loss(self):
        return self.__loss

    def accuracy(self):
        return self.__accuracy
