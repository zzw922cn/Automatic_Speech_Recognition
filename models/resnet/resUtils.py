
import tensorflow as tf
import tensorflow.contrib.slim as slim



def resUnit(input_layer, unit_id, is_training=True):
    with tf.variable_scope('ResUnit_'+str(unit_id), initializer=tf.random_normal_initializer()):
        bn_layer1 = tf.contrib.layers.batch_norm(input_layer, is_training=is_training)
        relu_layer1 = tf.nn.relu(bn_layer1)
        conv_layer1 = tf.layers.conv2d(relu_layer1, 32, (3,3), padding='same')
        bn_layer2 = tf.contrib.layers.batch_norm(conv_layer1, is_training=is_training)
        relu_layer2 = tf.nn.relu(bn_layer2)
        conv_layer2 = tf.layers.conv2d(relu_layer2, 32, (3,3), padding='same')
    return input_layer+conv_layer2

def highwayUnit(input_layer, unit_id, is_training=True):
    with tf.variable_scope('HighwayUnit_'+str(unit_id), initializer=tf.random_normal_initializer()):
        T = tf.layers.conv2d(input_layer, 32, (3,3), padding='same')

        bn_layer1 = tf.contrib.layers.batch_norm(input_layer, is_training=is_training)
        relu_layer1 = tf.nn.relu(bn_layer1)
        conv_layer1 = tf.layers.conv2d(relu_layer1, 32, (3,3), padding='same')
        bn_layer2 = tf.contrib.layers.batch_norm(conv_layer1, is_training=is_training)
        relu_layer2 = tf.nn.relu(bn_layer2)
        conv_layer2 = tf.layers.conv2d(relu_layer2, 32, (3,3), padding='same')
    return (1.0-T)*input_layer+T*conv_layer2


if __name__ == '__main__':
    with tf.Session() as sess:
        input_layer = tf.get_variable('input', shape=[4,10,10,32], dtype=tf.float32)
        #out = resUnit(input_layer, 1)
        out = highwayUnit(input_layer, 1)
        sess.run(tf.global_variables_initializer())
        print sess.run(out)
