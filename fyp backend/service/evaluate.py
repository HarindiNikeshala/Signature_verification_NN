import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.csv_features import get_csv_features
from utils.load import load_model_from_checkpoint
from utils.multiplayer import multilayer_perceptron
from utils.read_csv import readCSV

learning_rate = 0.001
training_epochs = 1000

def evaluate_signature_from_image_local(image_path, user_id):
    test_path = f"data/testing/testing_{user_id}.csv"
    train_path = f"data/training/training_{user_id}.csv"

    print(f"Evaluating signature for user {user_id} using image: {image_path}")
    print(f"Test path: {test_path}")
    print(f"Train path: {train_path}")

    # Convert image to feature row
    # features = get_csv_features(image_path)
    # with open(test_path, 'w') as f:
    #     f.write(','.join(map(str, features)) + '\n')

     # 1) Extract raw features
    features = get_csv_features(image_path)
    # 2) Cast every element to a built-in Python float
    features = [float(x) for x in features]

    # 3) Write a clean, comma-separated row of numbers
    with open(test_path, 'w') as f:
        f.write(','.join(str(x) for x in features) + '\n')

    # Read train/test features
    train_input, corr_train, test_input = readCSV(train_path, test_path, type2=True)

    # If test_input is a 1D list, convert it to a 2D array with one row
    if len(test_input) == 0 or len(test_input[0]) == 0:
        raise ValueError("Test input is empty or malformed. Check your CSV/data conversion pipeline.")

    # Ensure test_input is properly shaped
    if len(test_input) == 1:
        test_input = [test_input[0]]

    sess, graph = load_model_from_checkpoint()

    with graph.as_default():
        with sess.as_default():
            X = graph.get_tensor_by_name("Placeholder:0")
            Y = graph.get_tensor_by_name("Placeholder_1:0")

            logits = multilayer_perceptron(X, graph)
            pred = tf.nn.softmax(logits, name="pred")
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

            loss_op = tf.reduce_mean(tf.squared_difference(logits, Y), name="loss_op")
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op, name="train_op")

            # Init uninitialized variables
            uninitialized_vars = [v for v in tf.global_variables() if not sess.run(tf.is_variable_initialized(v))]
            if uninitialized_vars:
                sess.run(tf.variables_initializer(uninitialized_vars))

            for epoch in range(training_epochs):
                _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
                if cost < 0.0001:
                    break

            train_accuracy = sess.run(accuracy, feed_dict={X: train_input, Y: corr_train})
            prediction = sess.run(pred, feed_dict={X: test_input})

            if prediction.shape[0] == 0:
                raise ValueError("Prediction failed: No output generated from model.")

            forged_conf = float(prediction[0][0] * 100)
            genuine_conf = float(prediction[0][1] * 100)

            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Confidence (Forged): {forged_conf:.2f}%")
            print(f"Confidence (Genuine): {genuine_conf:.2f}%")

            return True if genuine_conf > forged_conf else False, genuine_conf, forged_conf
