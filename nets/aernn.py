from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def aernn_arg_scope(is_training = True, weight_decay=0.0005):
  """Defines the VAE arg scope.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer = slim.l2_regularizer(weight_decay),
                      #biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True,}
                      )as arg_sc:
    return arg_sc

def encoder(inputs, is_training=True, dropout_keep_prob=0.5):
  with tf.variable_scope('encoder') as sc:
    with slim.arg_scope([slim.conv2d], kernel_size = 3, padding = 'same'):
      print (inputs.shape)
      net = slim.repeat(inputs, 1, slim.conv2d, 32,  scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 64, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 128, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 256, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 512, scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      print (net.shape)
      net = slim.conv2d(net, 4096, kernel_size = [7, 7], padding = 'valid', scope='fc6')
      #net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      print (net.shape)  
      net = tf.squeeze(net, [1, 2])
      print (net.shape)       
      # Convert end_points_collection into a end_point dict.
    return net

def decoder(inputs, is_training=True):
  with tf.variable_scope('decoder') as sc:
    with slim.arg_scope([slim.conv2d_transpose], kernel_size = 3, stride = 2, padding='same'):
      net = inputs
      print (net.shape, 'input')
      net = slim.repeat(net, 1, slim.conv2d_transpose, 512, kernel_size = [7, 7], stride = 1, padding = 'valid', scope='tran_conv1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d_transpose, 256, scope='tran_conv2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d_transpose, 128, scope='tran_conv3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d_transpose, 64, scope='tran_conv4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d_transpose, 32, scope='tran_conv5')  
      print (net.shape)
      if is_training:
        net = slim.repeat(net, 1, slim.conv2d_transpose, 3, activation_fn = None, normalizer_fn = None, scope='tran_conv6')
      else:
        net = slim.repeat(net, 1, slim.conv2d_transpose, 3, activation_fn = tf.nn.sigmoid, normalizer_fn = None, scope='tran_conv6')
      print (net.shape) 
    return net

def aernn(inputs, label, num_classes=50, is_training=True):
  with tf.variable_scope('aernn', reuse = tf.AUTO_REUSE) as sc:
    end_points = {}
    num_units = 512
    net = []
    for i in range(inputs.shape[1]):
      image = tf.split(value = inputs, axis = 1, num_or_size_splits = inputs.shape[1])[i]
      image = tf.squeeze(image, 1)
      net.append(encoder(image, is_training))
    net = tf.stack(net, 1) 
    print (net.shape, 'net')
    lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units)
    initial_state = lstm_layer.zero_state(inputs.shape[0], dtype=tf.float32)
    outputs,state=tf.nn.dynamic_rnn(lstm_layer, net, initial_state = initial_state, dtype="float32")
    print (outputs.shape, 'outputs')
    net = slim.flatten(outputs)
    net = slim.fully_connected(net, 2048, activation_fn=None, normalizer_fn=None)
    end_points[sc.name + '/fc'] = net
    if is_training:
      rec_image = []
      net = tf.reshape(net, [8, 8, 256])
      for i in range(inputs.shape[1]):
        z = tf.split(value = net, axis = 1, num_or_size_splits = inputs.shape[1])[i]
        z = tf.expand_dims(z, 1)
        rec_image.append(decoder(z, is_training))
      net = tf.stack(rec_image, 1)

      rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = inputs, logits = net, reduction=tf.losses.Reduction.SUM)

      return net, end_points
    else:
      return net, end_points

aernn.default_image_size = 224


