import tensorflow as tf
import video_net_pre_function as function
import video_bottleneck_block as block


def create_network(input_tensor, fc_dim, keep_prob, numClass):
    num_channels = input_tensor.get_shape().as_list()[-1]
    res1 = function.new_conv_layer(input_tensor=input_tensor, layer_name='Res1', stride_time=1, stride=2, num_inChannel=num_channels,
                                   filter_size=4, num_filters=32, batch_norm=True, use_relu=True)
    print('res1: ')
    print(res1.get_shape())

    with tf.variable_scope('Res2'):
        res2_a = block.block(res1, 32, block_name='res2',
                             s1=1, k1=1, nf1=32, name1='res2a_branch_a',
                             s2=1, k2=3, nf2=32, name2='res2a_branch_b',
                             s3=1, k3=1, nf3=64, name3='res2a_branch_c',
                             s4=1, k4=1, name4='res2a_branch1', first_block=True)
        res2_b = block.block(res2_a, 64, block_name='res2',
                             s1=1, k1=1, nf1=32, name1='res2b_branch_a',
                             s2=1, k2=3, nf2=32, name2='res2b_branch_b',
                             s3=1, k3=1, nf3=64, name3='res2b_branch_c',
                             s4=1, k4=1, name4='res2b_branch2', first_block=False)
    print( 'res2: ' )
    print( res2_b.get_shape() )

    with tf.variable_scope('Res3'):
        res3_a = block.block(res2_b, 64, block_name='res3',
                             s1=1, k1=1, nf1=48, name1='res3a_branch_a',
                             s2=1, k2=3, nf2=48, name2='res3a_branch_b',
                             s3=1, k3=1, nf3=128, name3='res3a_branch_c',
                             s4=1, k4=1, name4='res3a_branch1', first_block=True)
        res3_b = block.block(res3_a, 128, block_name='res3',
                             s1=1, k1=1, nf1=48, name1='res3b_branch_a',
                             s2=1, k2=3, nf2=48, name2='res3b_branch_b',
                             s3=1, k3=1, nf3=128, name3='res3b_branch_c',
                             s4=1, k4=1, name4='res3b_branch2', first_block=False)
    print('res3: ')
    print( res3_b.get_shape() )

    with tf.variable_scope('Pool1'):
        pool1 = function.max_pool(res3_b, ksize=2, stride=2, name='Pool1')

    with tf.variable_scope('Res4'):
        res4_a = block.block(pool1, 128, block_name='res4',
                             s1=1, k1=1, nf1=48, name1='res4a_branch_a',
                             s2=1, k2=3, nf2=48, name2='res4a_branch_b',
                             s3=1, k3=1, nf3=256, name3='res4a_branch_c',
                             s4=1, k4=1, name4='res4a_branch1', first_block=True)
        res4_b = block.block(res4_a, 256, block_name='res4',
                             s1=1, k1=1, nf1=64, name1='res4b_branch_a',
                             s2=1, k2=3, nf2=64, name2='res4b_branch_b',
                             s3=1, k3=1, nf3=256, name3='res4b_branch_c',
                             s4=1, k4=1, name4='res4b_branch2', first_block=False)
    print( 'res4: ' )
    print( res4_b.get_shape() )

    with tf.variable_scope('Pool2'):
        pool2 = function.max_pool(res4_b, ksize=2, stride=2, name='Pool2')

    with tf.variable_scope('Res5'):
        res5_a = block.block(pool2, 256, block_name='res5',
                             s1=1, k1=1, nf1=128, name1='res5a_branch_a',
                             s2=1, k2=3, nf2=128, name2='res5a_branch_b',
                             s3=1, k3=1, nf3=512, name3='res5a_branch_c',
                             s4=1, k4=1, name4='res5a_branch1', first_block=True)
        res5_b = block.block(res5_a, 512, block_name='res5',
                             s1=1, k1=1, nf1=128, name1='res5b_branch_a',
                             s2=1, k2=3, nf2=128, name2='res5b_branch_b',
                             s3=1, k3=1, nf3=512, name3='res5b_branch_c',
                             s4=1, k4=1, name4='res5b_branch2', first_block=False)
    print( 'res5: ' )
    print( res5_b.get_shape() )

    with tf.variable_scope('Pool3'):
        pool3 = function.avg_pool(res5_b, ksize=4, stride=2, name='avg_pool')
    print( 'pool3: ' )
    print( pool3.get_shape() )

    with tf.variable_scope('Flatten'):
        flatten, _ = function.flatten_layer(pool3)
    print( 'flatten: ' )
    print( flatten.get_shape() )

    with tf.variable_scope('Fc1'):
        fc1 = function.fc_layer(flatten, fc_dim, name='Fc1', batch_norm=False, use_reg=True, use_relu=True)
    print( 'fc1: ' )
    print( fc1.get_shape() )

    with tf.variable_scope('Dropout'):
        dropout = function.dropout(fc1, keep_prob=keep_prob)
    print( 'dropout: ' )
    print( dropout.get_shape() )

    with tf.variable_scope('Softmax'):
        softmax = function.fc_layer(dropout, numClass, name='Softmax', batch_norm=False, use_reg=True, use_relu=False)
    print('softmax: ')
    print(softmax.get_shape())

    return softmax
