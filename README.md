
# RNN with matrix algebra

To me, RNNs are one of the most difficult deep learning approaches to gain an intuition for and to internalize how they work. 

In this repo I've written an implementation of a basic, single-layer RNN using low-level matrix operations. I'm using the [tf.Estimator API](https://www.tensorflow.org/guide/estimator) to structure the code and run the train, evaluate, and inference steps.

The estimator can be accessed by executing the `src/main.py` module (with `python3 src/main.py`). The repo uses a [weather dataset](https://www.bgc-jena.mpg.de/wetter/) created by the Max Planck Institute for Biogeochemistry. The estimator is trained to predict the change in air temperature over 10 minute increments. Running the `src/main.py` module yields the following error statistics:

```
Baseline error metrics (mean prediction) for training set: MAE=0.1639, R2=0.0
Baseline error metrics (mean prediction) for validation set: MAE=0.1604, R2=-2.2810e-07
Model error metrics for training set: MAE=0.1323, R2=0.2804
Model error metrics for validation set: MAE=0.1312, R2=0.2678
```

Here is a code excerpt of the model function for the single-layer RNN:

```python
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

# Define loss, mean absolute error
mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(labels, y_hat_t)))

# Optimizer and training objective
optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=params['learning_rate'])
train_op = optimizer.minimize(mae, global_step=tf.train.get_global_step())
```

(Variable notation taken from [Andrew Ng's Sequence Models course](https://www.coursera.org/learn/nlp-sequence-models/home/welcome))
