
# Import dependencies
import tensorflow as tf
import pandas as pd
import os


# Import weather data
def import_weather_data():

    # Download data as zip file
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
               'jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    return pd.read_csv(csv_path)


# Preprocess the weather data and split into train and validation
def preprocess_data(data):

    # Extract timestamp and temperature
    data = data.iloc[:, [0, 2]]
    data.columns = ['timestamp', 'temperature']
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Calculate the change in temperature
    data['previous_temp'] = data['temperature'].shift(1)
    data['temp_change'] = data['temperature'] - data['previous_temp']
    data = data.dropna(subset=['temp_change'])
    data = data.iloc[:, [0, 3]]

    # Split into train and validation sets
    train = data[data['timestamp'] < '2016-01-01 00:00:00']
    validation = data[data['timestamp'] >= '2016-01-01 00:00:00']

    return train, validation


# Write the preprocessed data to CSV files
def write_to_csv(train, validation, params):

    try:
        os.mkdir(os.path.dirname(params['train_file_pattern']))
    except:
        pass

    train.to_csv(params['train_file_pattern'], index=False)
    validation.to_csv(params['validation_file_pattern'], index=False)

    assert os.path.exists(params['train_file_pattern'])
    assert os.path.exists(params['validation_file_pattern'])


# Define input functions for training and evaluation
def input_fn(file_pattern, params, is_train=True):

    # Import CSV file into tf.data.Dataset object
    dataset = tf.data.experimental.CsvDataset(
        file_pattern, record_defaults=[tf.string, tf.float32], header=True)

    # Remove timestamp column
    dataset = dataset.map(lambda timestamp, temperature: temperature)

    # Create labels using the temperature change at the next time step
    labels_dataset = dataset.concatenate(
        tf.data.Dataset.from_tensor_slices([999.0]))
    dataset = tf.data.Dataset.from_tensor_slices([999.0]).concatenate(dataset)
    dataset = tf.data.Dataset.zip((dataset, labels_dataset))
    dataset = dataset.filter(
        lambda x, y: tf.math.not_equal(x, 999.0)
                     and tf.math.not_equal(y, 999.0))

    # Hard-code the batch size as 1 since the forward pass of the model
    # function depends on the hidden state from the previous example
    dataset = dataset.batch(1)
    # Repeat training dataset for desired number of epochs
    if is_train:
        dataset = dataset.repeat(params['num_epochs'])

    return dataset
