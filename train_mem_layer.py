r"""Script for training a model with the MemLayer.

python train_mem_layer.py --num_classes=2 \
  --batch_size=16 --memory_size=1024 --num_samples=1000 \
  --seed=0 --epochs=30 --num_features=10 --class_imb=0.9 \
  --n_clusters_per_class=5 --use_memory=True
"""

import tensorflow as tf

sess = tf.Session()

import keras.backend as K
K.set_session(sess)

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, ELU

import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from mem_layer import MemLayer


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_classes', 2, 'number of classes')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.flags.DEFINE_integer('mem_size', 2**10, 'number of slots in memory. '
                        'Leave as None to default to episode length')
tf.flags.DEFINE_integer('num_samples', 1000, 'number of training episodes')
tf.flags.DEFINE_integer('seed', 0, 'random seed for training sampling')
tf.flags.DEFINE_integer('epochs', 30, 'random seed for training sampling')
tf.flags.DEFINE_integer('num_features', 10, 'number of features')
tf.flags.DEFINE_float('class_imb', 0.9, 'class imbalance')
tf.flags.DEFINE_integer('clusters_per_class', 5, 'clusters per classes')
tf.flags.DEFINE_bool('use_memory', True, 'use memory')


def main(unused_argv):
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  x, y = make_classification(n_samples=FLAGS.num_samples, n_classes=FLAGS.num_classes,
                             weights=[FLAGS.class_imb], n_features=FLAGS.num_features,
                             n_redundant=0, n_informative=10,
                             n_clusters_per_class=FLAGS.clusters_per_class)

  x = StandardScaler().fit_transform(x)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)

  inp = tf.placeholder(tf.float32, shape=[None, FLAGS.num_features])
  labels = tf.placeholder(tf.int32, shape=[None])

  x = Dense(100)(inp)
  x = ELU()(x)
  x = Dense(100)(x)
  x = ELU()(x)

  if FLAGS.use_memory:
      ml = MemLayer(
        choose_k=256, mem_size=FLAGS.mem_size, num_classes=FLAGS.num_classes)
      closest_label, probs, teacher_loss = ml([x, labels])
      loss = teacher_loss + K.sparse_categorical_crossentropy(labels, probs)
  else:
      probs = Dense(FLAGS.num_classes, activation='softmax')(x)
      loss = K.sparse_categorical_crossentropy(labels, probs)

  preds = probs[:, 1]

  train_step = tf.train.AdamOptimizer().minimize(loss)

  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  batch_size = FLAGS.batch_size

  with sess.as_default():
      for epoch in range(FLAGS.epochs):
          for batch_i in range(x_train.shape[0] // batch_size):
              slc = slice(batch_i*batch_size, (batch_i+1)*batch_size)
              train_step.run(feed_dict={inp: x_train[slc],
                                        labels: y_train[slc],
                                        K.learning_phase(): 1})
          preds_ = np.zeros_like(y_test, dtype=np.float32)
          for batch_i in range(x_test.shape[0] // batch_size):
              slc = slice(batch_i*batch_size, (batch_i+1)*batch_size)
              preds_[slc] = preds.eval(feed_dict={inp: x_test[slc],
                                                  labels: y_test[slc],
                                                  K.learning_phase(): 0})
          pred_test = (preds_ > 0.5)
          acc = (y_test == pred_test).mean()
          auc = roc_auc_score(y_test, preds_)
          print('Epoch {}: acc={:.2f}, auc={:.2f}'.format(epoch, acc, auc))


if __name__ == '__main__':
  tf.app.run()
