# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : big_input.py
# Description  : Input pipeline for big dataset 
# ******************************************************

import tensorflow as tf
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import numpy as np
import os
import time
import math



# for any data type except int
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# for int data type
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## write tfrecord file
class RecordWriter(object):
  def __init__(self, path):
    self.path = path

  def write(self, content, filename, feature_num=2):
    tfrecords_filename = os.path.join(self.path, filename)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    if feature_num>1:
      assert isinstance(content, list), 'content must be a list now'
      feature_dict = {}
      for i in range(feature_num):
        feature = content[i]
        if isinstance(feature, int):
          feature_dict['feature'+str(i+1)]=_int64_feature(feature)
        else:
          feature_raw = np.array(feature).tostring()
          feature_dict['feature'+str(i+1)]=_bytes_feature(feature_raw)
      features_to_write = tf.train.Example(features=tf.train.Features(feature=feature_dict))
      writer.write(features_to_write.SerializeToString())
      writer.close()
      print('Record has been writen:'+tfrecords_filename)


## read tfrecord file
def read(filename_queue, feature_num=2, dtypes=[list, int]):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  feature_dict={}
  for i in range(feature_num):
    # here, only three data types are allowed: tf.float32, tf.int64, tf.string
    if dtypes[i] is int:
      feature_dict['feature'+str(i+1)]=tf.FixedLenFeature([], tf.int64)
    else:
      feature_dict['feature'+str(i+1)]=tf.FixedLenFeature([], tf.string)
  features = tf.parse_single_example(
      serialized_example,
      features=feature_dict)
  return features

#======================================================================================
## test code
flags.DEFINE_string("scale", "big", "specify your dataset scale")
flags.DEFINE_string("logdir", "/home/pony/github/data/inputpipeline/big", "specify the location to store log or model")
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
  tl = FLAGS.time_length
  fs = FLAGS.feature_size
  num_epochs = FLAGS.num_epochs
  batch_size = FLAGS.batch_size
  num_classes = FLAGS.num_classes
  num_batches = int(math.ceil(1.0*sn/batch_size))

  # x:[sn, tl, fs]
  with tf.variable_scope('train-samples'):
    x = []
    for n in range(sn):
      sub_x = np.random.rand(fs).astype(np.float32)
      x.append(sub_x)
       
  # y:[sn, tl]
  with tf.variable_scope('train-labels'):
    y = []
    for n in range(sn):
      sub_y = np.random.randint(0, num_classes)
      y_one_hot = np.eye(num_classes)[sub_y]
      y.append(y_one_hot.astype(np.int32))

  with tf.variable_scope('TFRecordWriter'):
    record_writer = RecordWriter(logdir)
    for n in range(sn):
      record_writer.write([x[n], y[n]], 'samples-labels[%s].tfrecords'%str(n))

  with tf.variable_scope('FilesProducer'):
    filenames = [os.path.join(logdir, 'samples-labels[%s].tfrecords' % str(i)) for i in range(sn)]
    filenamesQueue = tf.train.string_input_producer(filenames, num_epochs, shuffle=False)

  with tf.variable_scope('Reader'):
    features = read(filenamesQueue, dtypes=[list, list])
    # when handling array, must specify its shape, so reshape operation
    feature_x = tf.reshape(tf.decode_raw(features['feature1'], tf.float32), [fs])
    feature_y = tf.reshape(tf.decode_raw(features['feature2'], tf.int32), [num_classes])

  with tf.variable_scope('InputProducer'):
    batched_x, batched_y = tf.train.batch([feature_x, feature_y], batch_size=batch_size, dynamic_pad=False, allow_smaller_final_batch=True)
    batched_x = tf.layers.dense(batched_x, 2*fs)
    batched_x = tf.layers.dense(batched_x, num_classes)

  with tf.variable_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batched_y, logits=batched_x))
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
