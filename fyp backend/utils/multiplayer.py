import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
n_input = 9

# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
n_hidden_3 = 30 # 3rd layer
n_classes = 2 # no. of classes (genuine or forged)

# # tf Graph input
# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_classes])




def multilayer_perceptron(x, graph):
    with graph.as_default():
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_classes], seed=4))
        }
        layer_1 = tf.tanh(tf.matmul(x, weights['h1']) + biases['b1'])
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

