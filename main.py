#
# Recurrent autoencoder with sequence-aware encoding
# Source code of research (preprint):
# https://arxiv.org/pdf/2009.07349.pdf
#
# Author
# Robert Susik (rsusik@kis.p.lodz.pl)
# Institute of Applied Computer Science,
# Łódź University of Technology, Poland
# 2020
#
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from tqdm import tqdm
import shutil
import os
from glob import glob
import pickle
import itertools

from timeit import default_timer as timer
from datetime import datetime

import tensorflow as tf
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.saving import checkpoint_options as checkpoint_options_lib
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.distribute import distributed_file_utils
import six

tf.get_logger().setLevel("WARNING")

# Extending Tensorflow ModelCheckpoint
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/callbacks.py
#@keras_export('keras.callbacks.ModelCheckpoint')
class ModelCheckpoint(tf.keras.callbacks.Callback):
  """Callback to save the Keras model or model weights at some frequency.
  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.
  A few options this callback provides include:
  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.
  Example:
  ```python
  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_acc',
      mode='max',
      save_best_only=True)
  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```
  Arguments:
      filepath: string or `PathLike`, path to save the model file. `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, the latest best model according
        to the quantity monitored will not be overwritten.
        If `filepath` doesn't contain formatting options like `{epoch}` then
        `filepath` will be overwritten by each new better model.
      mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        overwrite the current save file is made based on either the maximization
        or the minimization of the monitored quantity. For `val_acc`, this
        should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               options=None,
               **kwargs):
    super(ModelCheckpoint, self).__init__()
    self.filepaths = []
    self._supports_tf_logs = True
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = tf.python.keras.utils.io_utils.path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.save_freq = save_freq
    self.epochs_since_last_save = 0
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0

    if save_weights_only:
      if options is None or isinstance(
          options, checkpoint_options_lib.CheckpointOptions):
        self._options = options or checkpoint_options_lib.CheckpointOptions()
      else:
        raise TypeError('If save_weights_only is True, then `options` must be'
                        'either None or a tf.train.CheckpointOptions')
    else:
      if options is None or isinstance(options, save_options_lib.SaveOptions):
        self._options = options or save_options_lib.SaveOptions()
      else:
        raise TypeError('If save_weights_only is False, then `options` must be'
                        'either None or a tf.saved_model.SaveOptions')

    # Deprecated field `load_weights_on_restart` is for loading the checkpoint
    # file from `filepath` at the start of `model.fit()`
    # TODO(rchao): Remove the arg during next breaking release.
    if 'load_weights_on_restart' in kwargs:
      self.load_weights_on_restart = kwargs['load_weights_on_restart']
      logging.warning('`load_weights_on_restart` argument is deprecated. '
                      'Please use `model.load_weights()` for loading weights '
                      'before the start of `model.fit()`.')
    else:
      self.load_weights_on_restart = False

    # Deprecated field `period` is for the number of epochs between which
    # the model is saved.
    if 'period' in kwargs:
      self.period = kwargs['period']
      logging.warning('`period` argument is deprecated. Please use `save_freq` '
                      'to specify the frequency in number of batches seen.')
    else:
      self.period = 1

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model
    # Use name matching rather than `isinstance` to avoid circular dependencies.
    if (not self.save_weights_only and
        not model._is_graph_network and  # pylint: disable=protected-access
        model.__class__.__name__ != 'Sequential'):
      self.save_weights_only = True

  def on_train_begin(self, logs=None):
    if self.load_weights_on_restart:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError('Error loading file from {}. Reason: {}'.format(
              filepath_to_load, e))

  def _implements_train_batch_hooks(self):
    # Only call batch hooks when saving on batch
    return self.save_freq != 'epoch'

  def on_train_batch_end(self, batch, logs=None):
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      self._save_model(epoch=epoch, logs=logs)

  def _should_save_on_batch(self, batch):
    """Handles batch-level saving logic, supports steps_per_execution."""
    if self.save_freq == 'epoch':
      return False

    if batch <= self._last_batch_seen:  # New epoch.
      add_batches = batch + 1  # batches are zero-indexed.
    else:
      add_batches = batch - self._last_batch_seen
    self._batches_seen_since_last_saving += add_batches
    self._last_batch_seen = batch

    if self._batches_seen_since_last_saving >= self.save_freq:
      self._batches_seen_since_last_saving = 0
      return True
    return False

  def _save_model(self, epoch, logs):
    """Saves the model.
    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best, current, filepath))
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                    filepath, overwrite=True, options=self._options)
              else:
                self.model.save(filepath, overwrite=True, options=self._options)
            else:
              if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, self.monitor, self.best))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          if self.save_weights_only:
            self.model.save_weights(
                filepath, overwrite=True, options=self._options)
          else:
            self.model.save(filepath, overwrite=True, options=self._options)

        self._maybe_remove_file()
        self.filepaths.append(filepath)
      except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in six.ensure_str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}` and
      # `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      file_path = self.filepath.format(epoch=epoch + 1, **logs)
    except KeyError as e:
      raise KeyError('Failed to format this callback filepath: "{}". '
                     'Reason: {}'.format(self.filepath, e))
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return file_io.file_exists_v2(filepath)
    tf_saved_model_exists = file_io.file_exists_v2(filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.
    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.
    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.
    Modified time of a file is obtained with `os.path.getmtime()`.
    This utility function is best demonstrated via an example:
    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```
    Arguments:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.
    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists_v2(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name

class Models(): 
    def RAESC(hidden_dim, seq_len, n_features):
        in1 = tf.keras.layers.Input(shape=(seq_len, n_features))
        gru1 = tf.keras.layers.GRU(hidden_dim, name='latent_layer')(in1)
        rsh1 = tf.keras.layers.Reshape( (hidden_dim, 1) )(gru1)
        cov1 = tf.keras.layers.Conv1D(seq_len, 3, padding='same')(rsh1)
        max1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(cov1)
        rsh2 = tf.keras.layers.Reshape( (seq_len, hidden_dim) )(max1)
        gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(rsh2)
        tdd1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(gru2)

        return tf.keras.Model(
            inputs=[in1], 
            outputs=[tdd1],
            name='RAESC'
        )

    def RAE(hidden_dim, seq_len, n_features):
        in1 = tf.keras.layers.Input(shape=(None, n_features))
        gru1 = tf.keras.layers.GRU(hidden_dim, name='latent_layer')(in1)
        rv1 = tf.keras.layers.RepeatVector(seq_len)(gru1)
        gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(rv1)
        tdd1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(gru2)

        return tf.keras.Model(
            inputs=[in1], 
            outputs=[tdd1],
            name='RAE'
        )

    def not_autoencoder(hidden_dim, seq_len, n_features):
        in1 = tf.keras.layers.Input(shape=(None, n_features))
        gru1 = tf.keras.layers.GRU(hidden_dim, name='latent_layer', return_sequences=True)(in1) # RETURN SEQUENCE
        gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(gru1)
        tdd1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(gru2)

        return tf.keras.Model(
            inputs=[in1], 
            outputs=[tdd1],
            name='not_autoencoder'
        )

    def RAES(hidden_dim, seq_len, n_features):
        if int(hidden_dim/seq_len) == 0 or hidden_dim % seq_len != 0:
            return None
        in1 = tf.keras.layers.Input(shape=(seq_len, n_features))
        gru1 = tf.keras.layers.GRU(hidden_dim, name='latent_layer')(in1)
        rsh2 = tf.keras.layers.Reshape( (seq_len, int(hidden_dim/seq_len)) )(gru1) # drugi wymiar / seq_len poniewaz seq_len < hidden_dim, wiec bylaby strata
        gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(rsh2)
        tdd1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(gru2)

        return tf.keras.Model(
            inputs=[in1], 
            outputs=[tdd1],
            name='RAES'
        )

    def RAECS_T(hidden_dim, seq_len, n_features):
        in1 = tf.keras.layers.Input(shape=(seq_len, n_features))
        gru1 = tf.keras.layers.GRU(hidden_dim, name='latent_layer')(in1)
        rsh1 = tf.keras.layers.Reshape( (hidden_dim, 1) )(gru1)
        cov1 = tf.keras.layers.Conv1DTranspose(seq_len, 3, padding='same')(rsh1)
        max1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(cov1)
        gru2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(max1)
        tdd1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(gru2)

        return tf.keras.Model(
            inputs=[in1], 
            outputs=[tdd1],
            name='RAECS_T'
        )


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


def train_model(
    model, 
    x, 
    x_val, 
    epochs, 
    batch_size, 
    log_dir, 
    learning_rate, 
    loss,
    **kwargs
):
    if 'latent_layer' not in [x.name for x in model.layers]:
        raise Exception('ERROR: The model does not have layer called "latent_layer"')

    print(f'Training: {model.name}, e-{epochs}, batch-{batch_size}, lr-{learning_rate}, loss-{loss}')
    log_dir = f'{log_dir}/{model.name}'
    
    os.makedirs(log_dir, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=opt)
    
    callbacks = []
    checkpointCallback = None
    time_counter = TimingCallback()
    callbacks.append(time_counter)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    checkpoint_paths = f'{log_dir}/checkpoints'
    os.makedirs(checkpoint_paths, exist_ok=True)
    checkpoint_paths += '/model.{epoch:02d}-{val_loss:.5f}.h5'
    checkpointCallback = ModelCheckpoint(
            filepath=checkpoint_paths, 
            save_weights_only=False, 
            monitor='val_loss', 
            mode='min', 
            save_best_only=False, 
            verbose=0
    )
    callbacks.append(
        checkpointCallback
    )

    history = model.fit(
        x, x, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_data=(x_val, x_val),
        callbacks=callbacks
    )

    if checkpointCallback != None:
        checkpoint_filepaths = checkpointCallback.filepaths
    else:
        checkpoint_filepaths = []

    return {
        'model': model,
        'checkpoints': checkpoint_filepaths,
        'history': history,
        'timings': time_counter.logs
    }


class TRAINING_DETAILS():
    NAME = 0
    EPOCHS = 1
    BATCH_SIZE = 2
    SEQ_LEN = 3
    FEATURES = 4
    HIDDEN_DIM = 5
    LEARNING_RATE = 6
    DR_RATE = 7
    LOSS = 8
    SAMPLES = 9
    HISTORY = 10
    TIMINGS = 11
    CHECKPOINTS = 12

def evaluate(
    results,
    output_path = '.'
):
    eval_results = []
    for result in results:

        with open(PATHS.get_dataset_path(
            output_path, 
            result[TRAINING_DETAILS.FEATURES], 
            result[TRAINING_DETAILS.SAMPLES]
        ), 'rb') as f:
            xx = pickle.load(f)
            x = xx['x']
            x_val = xx['x_val']
            del xx

        print(f'Evaluating: {result[TRAINING_DETAILS.NAME]}')
        log_dir = PATHS.get_result_path(
            output_path, 
            result[TRAINING_DETAILS.NAME],
            result[TRAINING_DETAILS.HIDDEN_DIM], 
            result[TRAINING_DETAILS.EPOCHS],
            result[TRAINING_DETAILS.BATCH_SIZE], 
            result[TRAINING_DETAILS.LEARNING_RATE],
            result[TRAINING_DETAILS.DR_RATE], 
            result[TRAINING_DETAILS.SAMPLES],
            result[TRAINING_DETAILS.FEATURES]
        )
        os.makedirs(log_dir, exist_ok=True)

        best_idx = np.argmin(result[TRAINING_DETAILS.HISTORY]['val_loss'])
        print(f'Best result for epoch: {best_idx+1}')
        
        model = tf.keras.models.load_model(result[TRAINING_DETAILS.CHECKPOINTS][best_idx])
        pred = model.predict(x_val)
        evaluation = model.evaluate(x_val, x_val)
        print(f'Result: {evaluation}')

        eval_results.append([
            evaluation,
            pred
        ])
    return eval_results


def save_results(
    train_results,
    eval_results,
    filename='./results.pickle'
):
    to_save = []
    for train_result, eval_result in tqdm(zip(train_results, eval_results)):
        to_save.append({
            'training': train_result,
            'evalution': eval_result
        })

    with open(f'{filename}', 'wb') as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    return to_save


class PATHS():
    @staticmethod
    def get_results_path(output_path):
        return f'{output_path}/results'

    @staticmethod
    def get_result_path(output_path, name, hidden_dim, epochs, 
        batch_size, learning_rate, dr_rate, samples, features
    ):
        results_path = PATHS.get_results_path(output_path)
        return f'{results_path}/exp_m-{name}_hd-{hidden_dim}_e-{epochs}_b-{batch_size}_l-{learning_rate}_d-{dr_rate}_s-{samples}_f-{features}'
    
    def get_datasets_path(output_path):
        return f'{output_path}/datasets'

    def get_dataset_path(output_path, features, samples):
        return f'{PATHS.get_datasets_path(output_path)}/dataset_w{features}_s{samples}.pickle'

def train(
    models,
    n_epochs = [10],
    n_hidden_dim_delimiter = [1], #[4, 2, 1]
    n_batch_size = [100],
    n_learning_rate = [0.001],
    n_dr_rate = [0.2],
    n_features = [1], #[1, 2, 4, 8]
    n_samples = [5000],
    output_path = '.'
):
    results_path = PATHS.get_results_path(output_path)
    results = []
    for (
        model, 
        hidden_dim_delimiter, 
        epochs, 
        batch_size, 
        learning_rate, 
        dr_rate, 
        samples, 
        features
    ) in tqdm(itertools.product(
        np.atleast_1d(models),
        np.atleast_1d(n_hidden_dim_delimiter),
        np.atleast_1d(n_epochs),
        np.atleast_1d(n_batch_size),
        np.atleast_1d(n_learning_rate),
        np.atleast_1d(n_dr_rate),
        np.atleast_1d(n_samples),
        np.atleast_1d(n_features)
    )):
        if not os.path.isfile(PATHS.get_dataset_path(output_path, features, samples)):
          raise Exception(f'ERROR: File does NOT exists: {PATHS.get_dataset_path(output_path, features, samples)}')
        with open(PATHS.get_dataset_path(output_path, features, samples), 'rb') as f:
            xx = pickle.load(f)
            x = xx['x']
            x_val = xx['x_val']
            del xx
        
        hidden_dim = int((x.shape[1]*features)/hidden_dim_delimiter)

        model = model(hidden_dim, x.shape[1], features)
        if model is None:
            print(f'ERROR: skipping as model is None')
            continue

        result_path = PATHS.get_result_path(output_path, model.name, hidden_dim, epochs, batch_size, learning_rate, dr_rate, samples, features)

        loss = 'mse'
        training_res = train_model( 
            model, 
            x=x, 
            x_val=x_val, 
            epochs=epochs, 
            batch_size=batch_size, 
            loss=loss,
            learning_rate=learning_rate,
            log_dir=result_path
        )
        
        results.append([
            model.name,
            epochs,
            batch_size,
            x.shape[1], #sequence length
            features,
            hidden_dim,
            learning_rate,
            dr_rate,
            str(loss),
            samples,
            training_res['history'].history,
            training_res['timings'],
            training_res['checkpoints']
        ])

    return results


def plot_results(
  data, 
  metric='loss', 
  filename=None, 
  label_fmt='{NAME}', 
  output_path='.', 
  line_styles_mapping=None,
  xticks=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
):
    mpl.use('pgf')
    mpl.rcParams["font.size"] = 14
    name_mapping = {
        'RAE': 'RAE',
        'RAESC': 'RAESC',
        'RAES': 'RAES'
    }

    color_mapping = {
        'RAE': ['red', 'firebrick', 'orange', 'salmon', 'gold', 'olive', 'y'],
        'RAESC': ['green'],
        'RAES': ['blue']
    }
    color_mapping = {key: itertools.cycle(color_mapping[key]) for key in color_mapping.keys()}

    if line_styles_mapping is None:
        line_styles_mapping = {
            'RAE': ['-'],#, '--', '-.', ':'],
            'RAESC': ['-'],
            'RAES': ['-']
        }
    line_styles_mapping = {key: itertools.cycle(line_styles_mapping[key]) for key in line_styles_mapping.keys()}
    
    plt.figure(figsize=(5, 3))
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.98, top=0.98)
    plt.gca().set_xlabel('epochs') # , fontsize=4
    plt.gca().set_ylabel(metric)
    if xticks is not None:
      plt.xticks(xticks, fontsize=9)
    plt.yticks(fontsize=9)
    for d in data:
        label = label_fmt\
            .replace('{NAME}', name_mapping[d['training'][TRAINING_DETAILS.NAME]])\
            .replace('{HD}', str(int(
                100*d['training'][TRAINING_DETAILS.HIDDEN_DIM] / (d['training'][TRAINING_DETAILS.SEQ_LEN] * d['training'][TRAINING_DETAILS.FEATURES])
            )).rjust(3, ' ')+'\%')\
            .replace('{SQLEN}', str(d['training'][TRAINING_DETAILS.SEQ_LEN]))\
            .replace('{FEAT}', str(d['training'][TRAINING_DETAILS.FEATURES]))\
            .replace('{SAMPL}', str(d['training'][TRAINING_DETAILS.SAMPLES]))\
            .replace('{BATCH}', str(d['training'][TRAINING_DETAILS.BATCH_SIZE]))
        plt.plot(d['training'][TRAINING_DETAILS.HISTORY][metric], label=label, c=next(color_mapping[d['training'][TRAINING_DETAILS.NAME]]), linestyle=next(line_styles_mapping[d['training'][TRAINING_DETAILS.NAME]]))
        
    handles, labels = plt.gca().get_legend_handles_labels()
    labels_order = np.argsort(np.array(labels))
    plt.legend(
        np.array(handles)[labels_order], np.array(labels)[labels_order], 
        prop = font_manager.FontProperties(size = 12)
    )
    if filename is not None:
        plt.savefig(f'{output_path}/{filename}')
        plt.savefig(f'{output_path}/{filename}.png')
