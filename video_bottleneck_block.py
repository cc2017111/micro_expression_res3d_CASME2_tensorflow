import tensorflow as tf
import video_net_pre_function as function


def block(input_tensor, num_channel, block_name,
                s1, k1, nf1, name1,
                s2, k2, nf2, name2,
                s3, k3, nf3, name3,
                s4, k4, name4, first_block=False):
    """Create a block"""
    with tf.variable_scope(block_name):
        # conv_1
        layer_conv1 = function.new_conv_layer(input_tensor, layer_name=name1, stride_time=s1, stride=s1,
                                              num_inChannel=num_channel, filter_size=k1,
                                              num_filters=nf1, batch_norm=True, use_relu=True)
        #conv_2
        layer_conv2 = function.new_conv_layer(layer_conv1, layer_name=name2, stride_time=s2, stride=s2,
                                              num_inChannel=nf1, filter_size=k2,
                                              num_filters=nf2, batch_norm=True, use_relu=True)
        #conv_3
        layer_conv3 = function.new_conv_layer(layer_conv2, layer_name=name3, stride_time=s3, stride=s3,
                                              num_inChannel=nf2, filter_size=k3,
                                              num_filters=nf3, batch_norm=True, use_relu=False)
        if first_block:
            shoutcut = function.new_conv_layer(input_tensor, layer_name=name4, stride_time=s4, stride=s4,
                                               num_inChannel=num_channel, filter_size=k4,
                                               num_filters=nf3, batch_norm=True, use_relu=False)
            assert (shoutcut.get_shape().as_list() == layer_conv3.get_shape().as_list())
            # Tensor sizes of the two branches are not matched!
            res = shoutcut + layer_conv3
        else:
            res = layer_conv3 + input_tensor
            assert (input_tensor.get_shape().as_list() == layer_conv3.get_shape().as_list())
            # Tensor sizes of the two branches are not matched!
    return tf.nn.relu(res)
