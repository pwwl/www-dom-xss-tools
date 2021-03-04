#!/usr/bin/env python

import argparse
import bz2 as bzip2
import collections
import glob
import gzip
import itertools
import json
import logging
import lzma
import math
import os
import sys

import yaml

import tensorflow as tf

from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance

# For the _Initializer
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.lookup_ops import TableInitializerBase

DEFAULT_N_FEATURES = 2**18
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_EPOCHS = 1
DEFAULT_SHUFFLE_SIZE = 16384
DEFAULT_DNN_HIDDEN_UNITS = (100, 100)
DEFAULT_DNN_EMBEDDING_SIZE = 8
DEFAULT_SAVE_SUMMARY_STEPS = 10
DEFAULT_LOG_STEP_COUNT = 100
DEFAULT_DNN_OPTIMIZER = 'Adagrad'
DEFAULT_LINEAR_OPTIMIZER = 'Ftrl'
DEFAULT_EVAL_THROTTLE_SECS = 600
DEFAULT_FIXED_LENGTH_BUFFER_SIZE = 1024

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

N_CLASSES = 2


SIZE_OF_INT32 = 4
OP_LIBRARY_NAME = "libword_bag_ops.so"

NAME_LINEAR_CLASSIFIER = 'linear_classifier'
NAME_DNN_LINEAR_CLASSIFIER = 'dnn_linear_classifier'
NAME_DNN_CLASSIFIER = 'dnn_classifier'
NAME_DNN_CUSTOM = 'custom_dnn_classifier'


_FILE_HANDLERS = {
  None : (lambda x: open(x, mode='r'), ''),
  '' : (lambda x: open(x, mode='r'), ''),
  'GZIP' : (lambda x: gzip.open(x, mode='rt'), '.gz'),
  'ZLIB' : (lambda x: gzip.open(x, mode='rt'), '.gz'),
  'LZMA' : (lambda x: lzma.open(x, mode='rt'), '.xz'),
  'BZIP2' : (lambda x: bzip2.open(x, mode='rt'), '.bz2')
}

def lines_from_files(fnames, compression_type=None):
  for filename in DatasetBuilder.glob_fnames(fnames):
    with _FILE_HANDLERS[compression_type][0](filename) as ifile:
      for line in ifile:
        yield line.rstrip('\n')



def builder_from_files(fnames, fmt_type='lines', compression_type=None):
  return _FMT_HANDLERS[fmt_type](fnames, compression_type=compression_type)


class DatasetBuilder(object):
  def build(self):
    raise NotImplementedError()

  def hooks(self):
    return []

  @staticmethod
  def glob_fnames(fnames):
    answer = []
    for fname in fnames:
      # Hack to support sharded worker dataset
      if isinstance(fname, tf.Tensor):
        return fnames

      glob_res = glob.glob(fname)
      if len(glob_res) == 0:
        raise ValueError('Cannot find files for: ' + fname)

      answer += glob_res

    logging.info('globbing files resolved to: %s', answer)
    return answer


class DatasetFromFiles(DatasetBuilder):
  def __init__(self, fnames, compression_type=None):
    self._fnames = fnames
    self._compression_type = compression_type

  @classmethod
  def new(cls, fnames, compression_type=None):
    return cls(fnames, compression_type=compression_type)

  def build(self):
    return tf.data.TextLineDataset(
      self.glob_fnames(self._fnames),
      compression_type=self._compression_type)


class ConcatenatingDatasetBuilder(DatasetBuilder):
  def __init__(self, dsets):
    self._dsets = dsets

  def hooks(self):
    return itertools.chain(*[dset.hooks() for dset in self._dsets])

  def build(self):
    iterator = iter(self._dsets)
    try:
      first_item = next(iterator)
    except StopIteration:
      return DatasetFromArray.from_strings([]).build()

    dset = first_item.build()
    while True:
      try:
        next_builder = next(iterator)
        dset = dset.concatenate(next_builder.build())
      except StopIteration:
        break

    return dset


# Copy of KeyValueTensorInitializer that does not add the initializer to the
# graph collection so that it can be manually initialized. Otherwise, the
# initializer will fail to find the placeholder because the monitored session
# does not know the placeholder's value. Using Scaffold won't work to pass the
# placeholder feed dict in because that can only be used from inside the
# estimator class, not for the input pipeline.
#
# from: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/ops/lookup_ops.py
class _Initializer(TableInitializerBase):
  def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
    with tf.name_scope(name, "key_value_init", [keys, values]) as scope:
      self._keys = tf.convert_to_tensor(keys, dtype=key_dtype, name="keys")
      self._values = tf.convert_to_tensor(
        values, dtype=value_dtype, name="values")
      self._name = scope

    super(_Initializer, self).__init__(self._keys.dtype, self._values.dtype)

  def initialize(self, table):
    with tf.name_scope(
        self._name, values=(table.table_ref, self._keys,
                            self._values)) as scope:
      if context.executing_eagerly():
        # Ensure a unique name when eager execution is enabled to avoid spurious
        # sharing issues.
        scope += str(ops.uid())

      init_op = gen_lookup_ops.initialize_table_v2(
        table.table_ref, self._keys, self._values, name=scope)

    return init_op


# Copy of the tensorflow method that uses _Initializer
def index_to_string_table_from_tensor(
    vocabulary_list, default_value='UNK', name=None):
  if vocabulary_list is None:
    raise ValueError("vocabulary_list must be specified.")

  with tf.name_scope(name, "index_to_string") as scope:
    vocabulary_list = tf.convert_to_tensor(vocabulary_list, tf.string)
    num_elements = array_ops.size(vocabulary_list)
    keys = math_ops.to_int64(math_ops.range(num_elements))

    shared_name = ""
    init = _Initializer(
      keys, vocabulary_list, tf.int64, tf.string, name="table_init")
    return tf.contrib.lookup.HashTable(
      init, default_value, shared_name=shared_name, name=scope)



class CompressedRepetitionDataset(DatasetBuilder):
  class _Hook(tf.train.SessionRunHook):
    def __init__(self, builder):
      self._builder = builder

    def after_create_session(self, session, _):
      session.run(
        self._builder._table.init, feed_dict={
          self._builder._placeholder : self._builder._data_elems
        })

      # Release the memory after we have used it
      self._builder._data_elems = None

  def __init__(self, idx_fname, data_elems, compression_type=None):
    self._compression_type = compression_type
    self._idx_fname = idx_fname
    self._data_elems = data_elems
    self._table = None
    self._placeholder = None

  @classmethod
  def new(cls, fnames, compression_type=None):
    builders = []
    for fname in fnames:
      data_fname = fname + '.data'
      if not os.path.exists(data_fname):
        with_extension = data_fname + _FILE_HANDLERS[compression_type][1]
        if os.path.exists(with_extension):
          logging.info(
            'Using compressed extension filename: %s', with_extension)
          data_fname = with_extension

        else:
          raise ValueError('Cannot find %s' % data_fname)

      idx_fname = fname + '.idx'
      if not os.path.exists(idx_fname):
        raise ValueError('Cannot find %s'% idx_fname)

      builders.append(
        cls.from_files(
          idx_fname, data_fname,
          idx_compression_type=None,
          data_compression_type=compression_type))

    return ConcatenatingDatasetBuilder(builders)

  @classmethod
  def from_files(cls, idx_fname, data_fname,
                 idx_compression_type=None, data_compression_type=None):
    logging.info(
      'reading compressed file into memory... this may take a while.')
    data = list(lines_from_files(
      [data_fname], compression_type=data_compression_type))
    logging.info('done reading compressed file into memory...')

    return cls(
      idx_fname,
      data,
      compression_type=idx_compression_type)

  def hooks(self):
    return [self._Hook(self)]

  def build(self):
    self._placeholder = tf.placeholder(
      tf.string, shape=[None], name='ph_lookup')

    self._table = index_to_string_table_from_tensor(
      self._placeholder,
      default_value='UNK',
      name='decompress_repetition')

    if self._compression_type is None:
      # Don't provide the compression type argument if not needed, so that older
      # versions of TF which don't support a compression type for
      # FixedLengthRecordDataset won't error out if we don't request a
      # compression type
      dset = tf.data.FixedLengthRecordDataset(
        [self._idx_fname],
        SIZE_OF_INT32,
        buffer_size=DEFAULT_FIXED_LENGTH_BUFFER_SIZE)
    else:
      dset = tf.data.FixedLengthRecordDataset(
        [self._idx_fname],
        SIZE_OF_INT32,
        buffer_size=DEFAULT_FIXED_LENGTH_BUFFER_SIZE,
        compression_type=self._compression_type)

    def _map_fn(idxes_as_bytes):
      # Must reshape because decode_raw may return a shape of (?,) even with a
      # scalar argument
      #
      # the lookup operation requires int64 dtype
      decoded = tf.cast(
        tf.reshape(tf.decode_raw(idxes_as_bytes, tf.int32), ()),
        dtype=tf.int64)

      return self._table.lookup(decoded)

    return dset.map(_map_fn)


class ShardedWorkerDataset(DatasetBuilder):
  def __init__(
      self, fnames, shard_num, shard_index,
      fmt_type='lines', batch_size=DEFAULT_BATCH_SIZE, compression_type=None):
    self._fnames = fnames
    self._compression_type = compression_type
    self._shard_num = shard_num
    self._shard_index = shard_index
    self._batch_size = batch_size
    self._fmt_type = fmt_type
    if len(fnames) < shard_num:
      raise ValueError('file names size must be >= shard_num')
    self._hooks = []

  def hooks(self):
    return self._hooks

  def build(self):
    def _dataset_from_filename(fname):
      builder = builder_from_files(
        [fname],
        fmt_type=self._fmt_type,
        compression_type=self._compression_type)
      self._hooks += builder.hooks()
      return builder.build()

    dset = tf.data.Dataset.from_tensor_slices(
      self.glob_fnames(self._fnames))
    dset = dset.shard(self._shard_num, self._shard_index)
    return dset.interleave(
      _dataset_from_filename,
      cycle_length=self._shard_num,
      block_length=self._batch_size)


class DatasetFromArray(DatasetBuilder):
  def __init__(self, items, dtypes, shapes=None):
    self.items = items
    self.dtypes = dtypes
    self.shapes = shapes

  def build(self):
    return tf.data.Dataset.from_generator(
      lambda: self.items, self.dtypes, self.shapes)

  @classmethod
  def from_strings(cls, items):
    # Need to encode the strings first because of a bug in TF version 1.4
    # https://stackoverflow.com/questions/47705684/tensorflow-tf-data-dataset-from-generator-does-not-work-with-strings-on-pyt
    return cls((x.encode('utf8') for x in items), dtypes=(tf.string))


_FMT_HANDLERS = {
  'lines' : DatasetFromFiles.new,
  'lines-cache': CompressedRepetitionDataset.new
}


class MiniBatcher(object):
  def __init__(self, iterable, batch_size):
    self._iterable = iterable
    self._batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    items = []
    for _ in range(self._batch_size):
      items.append(next(self._iterable))

    return items

  def next(self):
    return self.__next__()


class InputParser(object):
  ParseData = collections.namedtuple(
    'ParseData',
    ['label', 'weight', 'feature_index', 'feature_value',
     'feature_terms', 'feature_shape', 'url', 'injections', 'safe', 'unknown'])

  InputData = collections.namedtuple(
    'InputData',
    ['label', 'weight', 'feature_value', 'feature_terms', 'url',
     'inj', 'safe', 'unk'])

  def __init__(self, module_dir, op_name=OP_LIBRARY_NAME):
    self._lib = tf.load_op_library(os.path.join(module_dir, op_name))

  def read_instance(self, input_tensor):
    return self.ParseData(*self._lib.parse_json_word_bag(input_tensor))

  def convert_instance(self, input_tensor, positive_weight_ratio=1):
    raw_data = self.read_instance(input_tensor)
    label = raw_data.label
    raw_data_weights = raw_data.weight

    if positive_weight_ratio == 1:
      weights = raw_data_weights
    else:
      weights = tf.where(
        tf.equal(label, POSITIVE_LABEL),
        tf.multiply(raw_data_weights, positive_weight_ratio),
        raw_data_weights)

    return self.InputData(
      label=tf.expand_dims(label, axis=1),
      weight=weights,
      feature_value=tf.SparseTensor(
        raw_data.feature_index, raw_data.feature_value, raw_data.feature_shape),
      feature_terms=tf.SparseTensor(
        raw_data.feature_index, raw_data.feature_terms, raw_data.feature_shape),
      url=raw_data.url,
      inj=raw_data.injections,
      safe=raw_data.safe,
      unk=raw_data.unknown)


class InputFunctionCreator(object):
  def make_train_fn(self, train_dataset_builder):
    raise NotImplementedError()

  def make_test_fn(self, test_dataset_builder):
    raise NotImplementedError()


class EstimatorInputFunction(InputFunctionCreator):
  def __init__(self, model):
    self._model = model

  def make_train_fn(self, train_dataset_builder):
    return self._model.make_train_fn(train_dataset_builder)

  def make_test_fn(self, test_dataset_builder):
    return self._model.make_test_fn(test_dataset_builder)


class KerasInputFunction(InputFunctionCreator):
  def __init__(self, model, feature_columns):
    self._model = model
    self._feature_columns = feature_columns

  def _translate(self, dataset_fn):
    def _fn():
      inputs, labels = dataset_fn().make_one_shot_iterator().get_next()
      features = tf.feature_column.input_layer(inputs, self._feature_columns)
      return features, labels

    return _fn

  def make_train_fn(self, train_dataset_builder):
    return self._translate(self._model.make_train_fn(train_dataset_builder))

  def make_test_fn(self, test_dataset_builder):
    return self._translate(self._model.make_test_fn(test_dataset_builder))



class Model(object):
  CLASSIFIERS = [
    NAME_LINEAR_CLASSIFIER, NAME_DNN_LINEAR_CLASSIFIER, NAME_DNN_CLASSIFIER,
    NAME_DNN_CUSTOM]

  class Config(object):
    def __init__(
        self,
        n_features=DEFAULT_N_FEATURES,
        positive_weight_ratio=1,
        model_name=NAME_LINEAR_CLASSIFIER,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle_buffer_size=DEFAULT_SHUFFLE_SIZE,
        train_max_steps=None,
        test_max_steps=None,
        train_epochs=DEFAULT_TRAIN_EPOCHS,
        dnn_hidden_units=DEFAULT_DNN_HIDDEN_UNITS,
        dnn_embedding_size=DEFAULT_DNN_EMBEDDING_SIZE,
        save_summary_steps=DEFAULT_SAVE_SUMMARY_STEPS,
        log_step_count_steps=DEFAULT_LOG_STEP_COUNT,
        prefetch_amount=None,
        num_parallel_calls=None,
        linear_optimizer=DEFAULT_LINEAR_OPTIMIZER,
        dnn_optimizer=DEFAULT_DNN_OPTIMIZER,
        eval_throttle_secs=DEFAULT_EVAL_THROTTLE_SECS,
        custom_dnn_layers_trainable=None,
        extra_numeric_columns=None):
      self.n_features = n_features
      self.positive_weight_ratio = positive_weight_ratio
      self.model_name = model_name
      self.shuffle_buffer_size = shuffle_buffer_size
      self.batch_size = batch_size
      self.train_max_steps = train_max_steps
      self.test_max_steps = test_max_steps
      self.train_epochs = train_epochs
      self.dnn_hidden_units = dnn_hidden_units
      self.dnn_embedding_size = dnn_embedding_size
      self.save_summary_steps = save_summary_steps
      self.log_step_count_steps = log_step_count_steps
      self.prefetch_amount = prefetch_amount
      self.num_parallel_calls = num_parallel_calls
      self.linear_optimizer = linear_optimizer
      self.dnn_optimizer = dnn_optimizer
      self.eval_throttle_secs = eval_throttle_secs
      self.custom_dnn_layers_trainable = custom_dnn_layers_trainable
      self.extra_numeric_columns = extra_numeric_columns or []

  def __init__(self,
               input_parser,
               config=None,
               model_dir=None,
               multi_gpu=None,
               profiling=False):

    if config:
      if isinstance(config, dict):
        self.config = self.Config(**config)
      else:
        self.config = config

    self.input_parser = input_parser
    self._input_fn_creator = None
    self._multi_gpu = multi_gpu
    self._model_dir = model_dir
    self._profiling = profiling

  def build_model(self):
    model = self._build_model(self._model_dir, self._multi_gpu)

    def _metric_fn(features, labels, predictions):
      pred = tf.argmax(predictions['logits'], 1)
      weights = features['weight']
      return {
        'true_positives' : tf.metrics.true_positives(
          labels, pred, weights=weights),
        'false_negatives' : tf.metrics.false_negatives(
          labels, pred, weights=weights),
        'false_positives' : tf.metrics.false_positives(
          labels, pred, weights=weights),
        'true_negatives' : tf.metrics.true_negatives(
          labels, pred, weights=weights)
      }

    return tf.contrib.estimator.add_metrics(model, _metric_fn)

  def _build_model(self, model_dir, multi_gpu):
    weight_col = tf.feature_column.numeric_column('weight', dtype=tf.int64)

    train_distribute = None
    if multi_gpu is not None:
      train_distribute = tf.contrib.distribute.MirroredStrategy(
        num_gpus=multi_gpu)

    run_config = tf.estimator.RunConfig(
      train_distribute=train_distribute,
      save_summary_steps=self.config.save_summary_steps,
      log_step_count_steps=self.config.log_step_count_steps)

    model_name = self.config.model_name
    if model_name == NAME_LINEAR_CLASSIFIER:
      self._input_fn_creator = EstimatorInputFunction(self)
      return tf.estimator.LinearClassifier(
        feature_columns=self._make_linear_columns(),
        weight_column=weight_col,
        n_classes=N_CLASSES,
        model_dir=model_dir,
        config=run_config,
        optimizer=self._make_linear_optimizer())
    elif model_name == NAME_DNN_LINEAR_CLASSIFIER:
      self._input_fn_creator = EstimatorInputFunction(self)
      return tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=self._make_linear_columns(),
        dnn_feature_columns=self._make_dnn_columns(),
        weight_column=weight_col,
        n_classes=N_CLASSES,
        model_dir=model_dir,
        dnn_hidden_units=self.config.dnn_hidden_units,
        config=run_config,
        linear_optimizer=self._make_linear_optimizer(),
        dnn_optimizer=self._make_dnn_optimizer())
    elif model_name == NAME_DNN_CLASSIFIER:
      self._input_fn_creator = EstimatorInputFunction(self)
      return tf.estimator.DNNClassifier(
        hidden_units=self.config.dnn_hidden_units,
        feature_columns=self._make_dnn_columns(),
        model_dir=model_dir,
        weight_column=weight_col,
        n_classes=N_CLASSES,
        config=run_config,
        optimizer=self._make_dnn_optimizer())
    elif model_name == NAME_DNN_CUSTOM:
      self._input_fn_creator = EstimatorInputFunction(self)
      dnn_columns = self._make_dnn_columns()
      
      def _model_fn(features, labels, mode, params):
        net = tf.feature_column.input_layer(features, dnn_columns)
        trainable_config = self.config.custom_dnn_layers_trainable

        expected_trainable_size = len(self.config.dnn_hidden_units) + 1
        if trainable_config is None:
          trainable_config = [True] * expected_trainable_size
        elif len(trainable_config) != expected_trainable_size:
          raise ValueError(
            'expected len(custom_dnn_layers_trainable) == '
            'len(self.config.dnn_hidden_units) + 1.')

        for i, units in enumerate(self.config.dnn_hidden_units):
          net = tf.layers.dense(
            net,
            units=units,
            activation=tf.nn.relu,
            trainable=trainable_config[i])

        logits = tf.layers.dense(
          net,
          N_CLASSES,
          activation=None,
          trainable=trainable_config[len(self.config.dnn_hidden_units)])
        predicted_classes = tf.argmax(logits, 1)
        batch_size = tf.shape(logits)[0]
        repeat_for_batch = tf.stack([batch_size, 1])
        class_names = tf.tile(
          tf.convert_to_tensor(['0', '1'], dtype=tf.string)[tf.newaxis, :],
          repeat_for_batch)
        class_ids = tf.tile(
          tf.convert_to_tensor([0, 1], dtype=tf.int32)[tf.newaxis, :],
          repeat_for_batch)

        if mode == tf.estimator.ModeKeys.PREDICT:
          probs = tf.nn.softmax(logits)
          return tf.estimator.EstimatorSpec(
            mode,
            predictions={
              'class_ids': class_ids,
              'probabilities': tf.nn.softmax(logits),
              'logits': logits,
              'classes': class_names
            },
            export_outputs={
              'output' : tf.estimator.export.ClassificationOutput(
                scores=probs,
                classes=class_names
              )})

        loss = tf.losses.sparse_softmax_cross_entropy(
          labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        if mode == tf.estimator.ModeKeys.EVAL:
          probs = tf.nn.softmax(logits)
          return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=metrics,
            predictions={
              'class_ids': class_ids,
              'probabilities': probs,
              'logits': logits,
              'classes': class_names
            },
            export_outputs={
              'output' : tf.estimator.export.ClassificationOutput(
                scores=probs,
                classes=class_names
              )})

        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = get_optimizer_instance(
          self.config.dnn_optimizer,
          learning_rate=0.05)   # Use 0.05 because that is what is used in the
                                # current implementation by default for DNN
        train_op = optimizer.minimize(
          loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op)

      return tf.estimator.Estimator(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=run_config)

    else:
      raise ValueError('unknown model name %s' % model_name)

  def _make_linear_optimizer(self):
    return self.config.linear_optimizer

  def _make_dnn_optimizer(self):
    return self.config.dnn_optimizer

  def _make_weighted_feature_col(self):
    return tf.feature_column.weighted_categorical_column(
      categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
        key='feature_terms',
        hash_bucket_size=self.config.n_features),
      weight_feature_key='feature_value',
      dtype=tf.int64)

  def _make_extra_numeric_columns(self):
    return [
      tf.feature_column.numeric_column(name, dtype=tf.int32)
      for name in self.config.extra_numeric_columns]

  def _make_dnn_columns(self):
    return (
      [tf.feature_column.embedding_column(
        self._make_weighted_feature_col(),
        self.config.dnn_embedding_size)] +
      self._make_extra_numeric_columns())

  def _make_linear_columns(self):
    return (
      [self._make_weighted_feature_col()] + self._make_extra_numeric_columns())

  def _input_feeder(self, builder, epochs=DEFAULT_TRAIN_EPOCHS, is_train=True):
    dataset = builder.build()
    if is_train:
      dataset = dataset.shuffle(self.config.shuffle_buffer_size)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(self.config.batch_size)
    if is_train:
      positive_weight_ratio = self.config.positive_weight_ratio
    else:
      positive_weight_ratio = 1

    dataset = dataset.map(
      lambda x: self._map_dataset_fn(x, positive_weight_ratio),
      num_parallel_calls=self.config.num_parallel_calls)

    prefetch_amount = self.config.prefetch_amount
    if prefetch_amount:
      dataset = dataset.prefetch(prefetch_amount)

    return dataset

  def _map_dataset_fn(self, input_tensor, positive_weight_ratio):
    data = self.input_parser.convert_instance(
      input_tensor, positive_weight_ratio=positive_weight_ratio)._asdict()
    labels = data.pop('label')
    return data, labels

  def make_train_fn(self, train_dataset_builder):
    def _train_fn():
      return self._input_feeder(
        train_dataset_builder, epochs=self.config.train_epochs, is_train=True)

    return _train_fn

  def make_test_fn(self, test_dataset_builder):
    def _test_fn():
      return self._input_feeder(test_dataset_builder, epochs=1, is_train=False)

    return _test_fn

  def train_and_test(self, train_data, test_data):
    
    if self._profiling:
      profile_hook = [tf.train.ProfilerHook(
        save_steps=20,
        output_dir=os.path.join(self._model_dir, "tracing"),
        show_dataflow=True,
        show_memory=True)]
    else:
      profile_hook = []

    return tf.estimator.train_and_evaluate(
      self.build_model(),
      train_spec=tf.estimator.TrainSpec(
        input_fn=self._input_fn_creator.make_train_fn(train_data),
        max_steps=self.config.train_max_steps,
        hooks=train_data.hooks()),
      eval_spec=tf.estimator.EvalSpec(
        input_fn=self._input_fn_creator.make_test_fn(test_data),
        steps=self.config.test_max_steps,
        throttle_secs=self.config.eval_throttle_secs,
        hooks=test_data.hooks()))

  def train(self, train_data):
    
    if self._profiling:
      profile_hook = [tf.train.ProfilerHook(
        save_steps=20,
        output_dir=os.path.join(self._model_dir, "tracing"),
        show_dataflow=True,
        show_memory=True)]
    else:
      profile_hook = []

    model = self.build_model()
    return model.train(
      self._input_fn_creator.make_train_fn(train_data),
      hooks=train_data.hooks())

  def evaluate(self, test_data):
    model = self.build_model()
    return model.evaluate(
      self._input_fn_creator.make_test_fn(test_data),
      hooks=test_data.hooks())

  def predict(self, iterable):
    def _input_fn():
      input_tensor = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_fn')
      data, _ = self._map_dataset_fn(input_tensor, 1)
      return tf.estimator.export.ServingInputReceiver(
        data, {'inputs' : input_tensor})

    predictor = tf.contrib.predictor.from_estimator(
      self.build_model(), _input_fn)
    count = 0
    for batch in MiniBatcher(iterable, self.config.batch_size):
      count += len(batch)
      out = predictor({'inputs' : batch})
      assert len(out['classes']) == len(batch), out['classes']
      for input_item, clazz, score in zip(batch, out['classes'], out['scores']):
        class_ids = [
          int(clazz_name.decode('utf8'))
          if isinstance(clazz_name, bytes)
          else clazz_name
          for clazz_name in clazz]
        score_weight = dict([
          (class_id, score_val.item())
          for class_id, score_val in zip(class_ids, score)])
        assert (0 in score_weight) and (1 in score_weight), '%s' % score_weight
        yield json.loads(input_item), score_weight

    logging.info('Number predicted: %d', count)

  def write_predictions_to_file(self, iterable, ofile):
    for instance, predictions in self.predict(iterable):
      instance['pred'] = predictions
      ofile.write(json.dumps(instance))
      ofile.write('\n')


def setup_env(task_id, total_tasks, port):
  if task_id or total_tasks:
    if not (task_id and total_tasks):
      raise ValueError('Must specify both --task_id and --total_tasks')

    if task_id < 1 or task_id > total_tasks:
      raise ValueError(
        '--task_id (%d) must be >= 1 and <= --total_tasks (%d)' % (
          task_id, total_tasks))

    if task_id == 1:
      task_type = 'chief'
      task_index = 0
    else:
      task_type = 'worker'
      task_index = task_id - 2

    os.environ['TF_CONFIG'] = json.dumps({
      'cluster' : {
        'chief' : ['localhost:%d' % port],
        'worker' : [
          'localhost:%d' % (port + i) for i in range(1, total_tasks)]
      },
      'task' : {'type' : task_type, 'index' : task_index}
    })



def main(args):
  log_format = (
    '%(levelname)s %(asctime)-15s %(filename)s:%(lineno)d: %(message)s')
  logging.basicConfig(
    format=log_format, level=getattr(logging, args['log_level']))
  tf.logging.set_verbosity(getattr(tf.logging, args['tf_log_level']))
  logging.info('Beginning. Called with args: %s', json.dumps(args, indent=4))
  tf.keras.backend.set_floatx('float16')
  logging.info('Set default float type to %s', tf.keras.backend.floatx())
  task_id, total_tasks = args['task_id'], args['total_tasks']
  setup_env(task_id=task_id, total_tasks=total_tasks, port=args['port'])

  model = Model(
    InputParser(args['c_module_dir']),
    config=Model.Config(
      model_name=args['classifier_name'],
      n_features=args['n_features'],
      positive_weight_ratio=args['positive_weight_ratio'],
      batch_size=args['batch_size'],
      shuffle_buffer_size=args['shuffle_buffer_size'],
      train_max_steps=args['train_max_steps'],
      test_max_steps=args['test_max_steps'],
      train_epochs=args['train_epochs'],
      dnn_hidden_units=args['dnn_hidden_units'],
      dnn_embedding_size=args['dnn_embedding_size'],
      save_summary_steps=args['save_summary_steps'],
      log_step_count_steps=args['log_step_count_steps'],
      prefetch_amount=args['prefetch_amount'],
      dnn_optimizer=args['dnn_optimizer'],
      linear_optimizer=args['linear_optimizer'],
      eval_throttle_secs=args['eval_throttle_secs'],
      custom_dnn_layers_trainable=args['custom_dnn_layers_trainable'],
      extra_numeric_columns=args['extra_numeric_columns']),
      model_dir=args['model_dir'],
      multi_gpu=args['multi_gpu'],
      profiling=args['profiling']
    )

  training_data = args['training_data']
  testing_data = args['testing_data']
  predict_data = args['predict_data']
  comp_type = args['compression_type']
  output_file_name = args['output_file']
  fmt_type = args['file_format']
  if args['use_train_and_evaluate']:
    if total_tasks and total_tasks > 1:
      train_dset = ShardedWorkerDataset(
        training_data, total_tasks, task_id - 1,
        compression_type=comp_type, fmt_type=fmt_type)
      test_dset = ShardedWorkerDataset(
        testing_data, total_tasks, task_id - 1,
        compression_type=comp_type, fmt_type=fmt_type)
    else:
      train_dset = builder_from_files(
        training_data,
        fmt_type=fmt_type,
        compression_type=comp_type)
      test_dset = builder_from_files(
        testing_data,
        fmt_type=fmt_type,
        compression_type=comp_type)

    model.train_and_test(train_dset, test_dset)

  else:
    if training_data:
      model.train(
        builder_from_files(
          training_data,
          fmt_type=fmt_type,
          compression_type=comp_type))

    if testing_data:
      logging.info(
        'metrics: %s',
        model.evaluate(
          builder_from_files(
            testing_data,
            fmt_type=fmt_type,
            compression_type=comp_type)))

  if predict_data:
    input_iterable = lines_from_files(predict_data, compression_type=comp_type)
    if output_file_name:
      with open(output_file_name, 'w') as ofile:
        model.write_predictions_to_file(input_iterable, ofile)

    else:
      model.write_predictions_to_file(input_iterable, sys.stdout)

  logging.info('Done')


def parser(parent=None):
  args = argparse.ArgumentParser(
    parents=[parent] if parent else [],
    description="""
Train a word bag document classifier. Input format is json-lines where each json
object should have the keys: 'lbl' (the label, either 'n' or 'p'), 'feat' (dict
of word counts, the features), and 'wght' (float, the weight).""")
  args.add_argument(
    '-t', '--training_data', nargs='+', help='Paths to training data files',
    default=[])
  args.add_argument(
    '-e', '--testing_data', nargs='+', help='Paths to testing data files',
    default=[])
  args.add_argument(
    '-p', '--predict_data', nargs='+', help='Paths to prediction files',
    default=[])
  args.add_argument(
    '-o', '--output_file', help='Output file for predictions')
  args.add_argument(
    '-n', '--classifier_name', help='classifier name',
    choices=sorted(Model.CLASSIFIERS), default=NAME_LINEAR_CLASSIFIER)
  args.add_argument(
    '--shuffle_buffer_size', '--shuffle_size',
    help='size of buffer for shuffling', default=DEFAULT_SHUFFLE_SIZE, type=int)
  args.add_argument(
    '--n_features', help='number of features', type=int,
    default=DEFAULT_N_FEATURES)
  args.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
  args.add_argument(
    '--positive_weight_ratio', help='Amount to weight positives',
    default=1, type=int)
  args.add_argument(
    '--log_level', choices=['INFO', 'WARN', 'ERROR'], default='INFO')
  args.add_argument(
    '--tf_log_level', choices=['INFO', 'WARN', 'ERROR'], default='WARN')
  args.add_argument('--c_module_dir', help='Path to c library', default='./')
  args.add_argument('--model_dir', help='path to store model files')
  args.add_argument(
    '--train_epochs', type=int,
    help='number of epochs to train', default=DEFAULT_TRAIN_EPOCHS)
  args.add_argument(
    '--task_id', type=int,
    help=('Task id for parallel training. Should be between 1 and '
          'total_tasks, inclusive.'))
  args.add_argument('--total_tasks', type=int, help='Total number of tasks')
  args.add_argument(
    '--port', type=int, help='Starting port number for distributed training',
    default=2222)
  args.add_argument(
    '--train_max_steps', type=int,
    help=('Number of batches to use during training. If not provided, '
          'will train forever in distributed training. When not in distributed '
          'training, will train for the number of epochs provided. '))
  args.add_argument(
    '--test_max_steps', type=int,
    help='Number of batches to use during testing. ')
  args.add_argument(
    '--dnn_hidden_units', nargs='+', type=int,
    default=DEFAULT_DNN_HIDDEN_UNITS,
    help='List of sizes of hidden units for the dnn models')
  args.add_argument(
    '--dnn_embedding_size', type=int,
    help='size of embedding for the dnn model')
  args.add_argument(
    '--save_summary_steps', type=int, default=DEFAULT_SAVE_SUMMARY_STEPS,
    help='number of steps to save tensorboard output')
  args.add_argument(
    '--compression_type',
    choices=sorted(filter(lambda x: x is not None, _FILE_HANDLERS.keys())),
    default=None,
    help=('Compression type for input datasets. Applied to --training_data, '
          '--testing_data. Tensorflow datasets only supports GZIP and ZLIB; '
          'Other options may be used for formats that do not require '
          'tensorflow support. '))
  args.add_argument(
    '--multi_gpu', type=int, default=None,
    help='Number of gpus in multi gpu training. Must be > 1')
  args.add_argument(
    '--log_step_count_steps', type=int, default=DEFAULT_LOG_STEP_COUNT,
    help='number of steps to log step info')
  args.add_argument(
    '--use_train_and_evaluate', action='store_true',
    help=('use the train_and_evaluate API for concurrent training and '
          'evaluation. Required when using distributed training. '))
  args.add_argument(
    '--profiling', action='store_true',
    help=('Write tf.Estimator profiling data to model directory.'))
  args.add_argument(
    '--prefetch_amount', type=int,
    help='Number of items to prefetch in input queue.')
  args.add_argument(
    '--num_parallel_calls', type=int,
    help='Number of parallel calls to process items in input queue')
  args.add_argument(
    '--dnn_optimizer', default=DEFAULT_DNN_OPTIMIZER,
    choices=['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'],
    help='Optimizer used for DNN models')
  args.add_argument(
    '--linear_optimizer', default=DEFAULT_LINEAR_OPTIMIZER,
    choices=['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'],
    help='Optimizer for linear models')
  args.add_argument(
    '--eval_throttle_secs', default=DEFAULT_EVAL_THROTTLE_SECS, type=int,
    help='Number of seconds between evaluation when using train_and_evaluate.')
  args.add_argument(
    '--custom_dnn_layers_trainable', default=None, nargs='+',
    type=int, help='List of whether specific layers are trainable.')
  args.add_argument(
    '--extra_numeric_columns', default=None, nargs='+',
    help=('List of extra numeric columns to include. Options: '
          '["inj", "safe", "unk"]'))
  args.add_argument(
    '--file_format', default='lines',
    choices=sorted(_FMT_HANDLERS.keys()),
    help='File format. Default is json-lines. ')
  return args


def parse_args(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  cfg_arg_parser = argparse.ArgumentParser(add_help=False)
  cfg_arg_parser.add_argument(
    '--cfg', nargs='+',
    help=('Configuration file. File format is yaml. Arguments on command line '
          'override config file arguments. '))
  cfg_args, remaining_argv = cfg_arg_parser.parse_known_args(argv)

  # hack to  get the default keys just so that we can alert the user if they
  # have some other config options
  defaults = parser().parse_args([])
  arg_parser = parser(parent=cfg_arg_parser)

  cfg_files = cfg_args.cfg
  if cfg_files:
    cfg_args = {}
    for cfg_file in cfg_files:
      with open(cfg_file, 'r') as cfg_data:
        new_cfg_args = yaml.safe_load(cfg_data)
        for key in new_cfg_args:
          if key not in defaults:
            raise ValueError('unknown configuration option %s' % key)

        cfg_args.update(new_cfg_args)

    arg_parser.set_defaults(**cfg_args)

  args = vars(arg_parser.parse_args(remaining_argv))
  if 'cfg' in args:
    del args['cfg']

  return args


if __name__ == '__main__':
  main(parse_args())
