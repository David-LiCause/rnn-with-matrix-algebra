
# Import dependencies
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# Initialize parameters for single-layer RNN model
def get_parameters(params):

    # Initialize the previous hidden state as vector of zeros
    # TODO(David) Add hook to reset hidden state after each session
    a_prev = tf.compat.v1.get_variable(
        name='a_prev', shape=[params['num_hidden_units'], 1], dtype=tf.float32,
        initializer=tf.constant_initializer(value=0), trainable=False)

    # Randomly initialize the weight matrix for the input features
    W_ax = tf.compat.v1.get_variable(
        name='W_ax', shape=[params['num_hidden_units'], 1], dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        trainable=True)
    # Randomly initialize the weight matrix for the previous hidden state
    W_aa = tf.compat.v1.get_variable(
        name='W_aa',
        shape=[params['num_hidden_units'], params['num_hidden_units']],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        trainable=True)
    # Randomly initialize the weight matrix for the output
    W_ay = tf.compat.v1.get_variable(
        name='W_ay', shape=[1, params['num_hidden_units']], dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        trainable=True)
    # Randomly initialize the bias term for the hidden state update
    b_a = tf.compat.v1.get_variable(
        name='b_a', shape=[params['num_hidden_units'], 1], dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        trainable=True)
    # Randomly initialize the bias term for the output
    b_y = tf.compat.v1.get_variable(
        name='b_y', shape=[1, 1], dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        trainable=True)

    return a_prev, W_ax, W_aa, W_ay, b_a, b_y


# Define model function for basic, single-layer RNN
def model_fn(features, labels, mode, params):

    # Adjust shapes for features, labels
    features = tf.expand_dims(features, 1)
    if labels is not None:
        labels = tf.expand_dims(labels, 1)

    # Get parameter values
    a_prev, W_ax, W_aa, W_ay, b_a, b_y = get_parameters(params)

    # Calculate the hidden state for the current time step
    a_t = tf.math.tanh(
        tf.add_n([
            tf.linalg.matmul(W_ax, features),
            tf.linalg.matmul(W_aa, a_prev),
            b_a]))

    # Get predicted value for the next time step
    y_hat_t = tf.add_n([
            tf.linalg.matmul(W_ay, a_t),
            b_y])

    # Update the hidden state of the model, this value will be used in the next
    # training step
    a_prev = tf.identity(a_t)

    # Define the inference mode for the estimator
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y_hat': y_hat_t
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Define loss, mean absolute error
    mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(labels, y_hat_t)))

    # Define the evaluation mode for the estimator
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=mae)

    # Optimizer and training objective
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(mae, global_step=tf.train.get_global_step())

    assert mode == tf.estimator.ModeKeys.TRAIN
    return tf.estimator.EstimatorSpec(mode, loss=mae, train_op=train_op)


# Get inferences from trained estimator
def get_inferences(estimator, input_fn):

    y_hat = []
    predictor = estimator.predict(input_fn, yield_single_examples=False)

    for pred in predictor:
        y_hat.append(pred['y_hat'][0, 0])

    return np.array(y_hat)


# Calculate mean absolute error and R squared
def get_metrics(y_true, y_hat):

    mae = np.mean(np.abs(y_true - y_hat))
    r2 = r2_score(y_true, y_hat)

    return mae, r2


# Using the mean of the training labels, get baseline error metrics
def print_baseline_metrics(params):

    train = pd.read_csv(params['train_file_pattern'])
    validation = pd.read_csv(params['validation_file_pattern'])

    train_labels = train['temp_change'].values[1:]
    validation_labels = validation['temp_change'].values[1:]

    mean_train_label = np.mean(train_labels)

    baseline_train_mae, baseline_train_r2 = get_metrics(
        y_true=train_labels,
        y_hat=np.full(shape=train_labels.shape, fill_value=mean_train_label))
    baseline_val_mae, baseline_val_r2 = get_metrics(
        y_true=validation_labels,
        y_hat=np.full(
            shape=validation_labels.shape, fill_value=mean_train_label))

    print('Baseline error metrics (mean prediction) for training set: MAE={},'
          ' R2={}'.format(baseline_train_mae, baseline_train_r2))
    print('Baseline error metrics (mean prediction) for validation set: '
          'MAE={}, R2={}'.format(baseline_val_mae, baseline_val_r2))


# Calculate MAE and R squared for training and validation sets
def print_model_metrics(params, train_preds, validation_preds):

    train = pd.read_csv(params['train_file_pattern'])
    validation = pd.read_csv(params['validation_file_pattern'])

    train_labels = train['temp_change'].values[1:]
    validation_labels = validation['temp_change'].values[1:]

    train_mae, train_r2 = get_metrics(y_true=train_labels, y_hat=train_preds[:-1])
    val_mae, val_r2 = get_metrics(
        y_true=validation_labels, y_hat=validation_preds[:-1])

    print('Model error metrics for training set: MAE={}, R2={}'.format(
        train_mae, train_r2))
    print('Model error metrics for validation set: MAE={}, R2={}'.format(
        val_mae, val_r2))
