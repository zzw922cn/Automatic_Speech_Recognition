# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : small_input.py
# Description  : Input pipeline for small dataset
# ******************************************************

import tensorflow as tf
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import os
import time
import math
import numpy as np

flags.DEFINE_string("scale", "small", "specify your dataset scale")
flags.DEFINE_string("logdir", "/home/pony/github/data/inputpipeline/small", "specify the location to store log or model")
flags.DEFINE_integer("samples_num", 80, "specify your total number of samples")
flags.DEFINE_integer("time_length", 2, "specify max time length of sample")
flags.DEFINE_integer("feature_size", 2, "specify feature size of sample")
flags.DEFINE_integer("num_epochs", 100, "specify number of training epochs")
flags.DEFINE_integer("batch_size", 2, "specify batch size when training")
flags.DEFINE_integer("num_classes", 10, "specify number of output classes")
FLAGS = flags.FLAGS

if __name__ == '__main__':

  scale = FLAGS.scale
  logdir = FLAGS.logdir
  sn = FLAGS.samples_num
  fs = FLAGS.feature_size
  num_epochs = FLAGS.num_epochs
  batch_size = FLAGS.batch_size
  num_classes = FLAGS.num_classes
  num_batches = int(math.ceil(1.0*sn/batch_size))

  with tf.variable_scope('train-samples'):
    x = tf.constant(np.random.rand(sn, fs).astype(np.float32))

  with tf.variable_scope('train-labels'):
    indices = np.random.randint(0, num_classes, size=sn).astype(np.int32)
    y = tf.one_hot(indices, depth=num_classes,
                   on_value=1.0,
                   off_value=0.0,
                   axis=-1,
                   dtype=tf.float32)

  # dequeue ops
  with tf.variable_scope('InputProducer'):
    slice_x, slice_y = tf.train.slice_input_producer([x, y], 
        num_epochs = num_epochs, seed=22, 
        capacity=36, shuffle=True)

    batched_x, batched_y = tf.train.batch([slice_x, slice_y], 
        batch_size=batch_size, dynamic_pad=False, 
        allow_smaller_final_batch=True)

    batched_x = tf.layers.dense(batched_x, 2*fs)
    batched_x = tf.layers.dense(batched_x, num_classes)
  
  with tf.variable_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=batched_y, logits=batched_x))
    optimizer = tf.train.AdamOptimizer(0.1)
    train_op = optimizer.minimize(loss)
    tf.summary.scalar('Loss', loss)

  merged = tf.summary.merge_all()

  t1 = time.time()
  sess = tf.Session()
  checkpoint_path = os.path.join(logdir, scale+'_model')
  writer = tf.summary.FileWriter(logdir, sess.graph)
  sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
  coord = tf.train.Coordinator()
  threads = queue_runner_impl.start_queue_runners(sess=sess)
  saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
  saver.save(sess, checkpoint_path)
  for i in range(num_batches*num_epochs):
    l, _, summary = sess.run([loss, train_op, merged])
    writer.add_summary(summary, i)
    print 'batch '+str(i+1)+'/'+str(num_batches*num_epochs)+'\tLoss:'+str(l)
  writer.close()	
  coord.request_stop()
  coord.join(threads)
  print 'program takes time:'+str(time.time()-t1)
