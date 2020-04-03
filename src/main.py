
# Import dependencies
import tensorflow as tf  # Version 1.15

# Import modules
import model
import input_data

# Set training hyperparameters
PARAMS = {
    'train_file_pattern': 'data/train.csv',
    'validation_file_pattern': 'data/validation.csv',
    'num_epochs': 1,
    'num_hidden_units': 16,
    'learning_rate': .001,
}

# Print TF training logs
tf.get_logger().setLevel('INFO')


# Train single-layer RNN model, print error metrics
def main():

    # Import and preprocess data
    weather_data = input_data.import_weather_data()
    train, validation = input_data.preprocess_data(weather_data)
    input_data.write_to_csv(train, validation, PARAMS)

    # Instantiate estimator
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, params=PARAMS)

    # Define train and eval input functions
    train_input_fn = lambda: input_data.input_fn(
        PARAMS['train_file_pattern'], PARAMS, is_train=True)
    eval_input_fn = lambda: input_data.input_fn(
        PARAMS['validation_file_pattern'], PARAMS, is_train=False)

    # Creae train_spec, eval_spec
    train_spec = tf.estimator.TrainSpec(train_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)

    # Run training and evaluation
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Generate inferences from estimator
    train_preds = model.get_inferences(estimator, train_input_fn)
    validation_preds = model.get_inferences(estimator, eval_input_fn)

    # Compare the single-layer RNN to a baseline using MAE and R squared
    model.print_baseline_metrics(PARAMS)
    model.print_model_metrics(PARAMS, train_preds, validation_preds)


if __name__ == '__main__':
    main()
