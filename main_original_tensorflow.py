# https://github.com/KennCoder7/HAR-CNN-1d
# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from sklearn import metrics
import data_file as df
import data_load as dl
from tensorflow.python.keras.utils.np_utils import to_categorical
import config as cfg
import util

# find data path
X_train_signals_paths = df.X_train_signals_paths
X_test_signals_paths = df.X_test_signals_paths
y_train_path = df.y_train_path
y_test_path = df.y_test_path

# load data
X_train = dl.load_X(X_train_signals_paths)
# print('X_train shape', X_train.shape)
X_test = dl.load_X(X_test_signals_paths)
# print('X_test shape', X_test.shape)
y_train = dl.load_y(y_train_path)
# y_train = to_categorical(y_train)
# print('y_train shape', y_train.shape)
y_test = dl.load_y(y_test_path)
# y_test = to_categorical(y_test)
# print('y_test shape', y_test.shape)



# Input Data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training



def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def LSTM_RNN(_X):
    # Graph weights
    _weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    _biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def CNN(X):
    # define
    seg_len = 128
    num_channels = 9
    num_labels = 6
    batch_size = 100
    learning_rate = 0.001
    num_epoches = 10000
    # training = tf.placeholder_with_default(False, shape=())
    # X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
    # Y = tf.placeholder(tf.float32, (None, num_labels))

    # CNN
    # convolution layer 1
    conv1 = tf.layers.conv1d(
        inputs=X,
        filters=64,
        kernel_size=3,
        activation=tf.nn.relu
    )
    print("### convolution layer 1 shape: ", conv1.shape, " ###")

    # convolution layer 2
    conv2 = tf.layers.conv1d(
        inputs=conv1,
        filters=64,
        kernel_size=3,
        activation=tf.nn.relu
    )
    print("### convolution layer 2 shape: ", conv2.shape, " ###")

    conv2 = tf.nn.dropout(conv2, keep_prob=0.5)
    print("### dropout layer shape: ", conv2.shape, " ###")

    # pooling layer
    pool= tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer shape: ", pool.shape, " ###")

    # flat
    l_op = pool
    shape = l_op.get_shape().as_list()
    flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
    print("### flat shape: ", flat.shape, " ###")

    # fully connected layer 1
    fc1 = tf.layers.dense(
        inputs=flat,
        units=100,
        activation=tf.nn.relu
    )
    print("### fully connected layer 1 shape: ", fc1.shape, " ###")


    # fully connected layer 3
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=num_labels,
        activation=tf.nn.softmax
    )
    print("### fully connected layer 2 shape: ", fc2.shape, " ###")

    # prediction
    # y_ = tf.layers.batch_normalization(fc3, training=training)
    y_ = fc2
    print("### prediction shape: ", y_.get_shape(), " ###")
    return y_


# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# pred = LSTM_RNN(x)
pred = CNN(x)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = util.extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(util.extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step * batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

# graph = tf.Graph()

# with graph.as_default():
#     # define
#     lstm_size =27
#     seg_len = 128
#     num_channels = 9
#     num_labels = 6
#     batch_size = 100
#     learning_rate = 0.001
#     num_epoches = 10000
#     lstm_layers = 2
#     training = tf.placeholder_with_default(False, shape=())
#     X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
#     Y = tf.placeholder(tf.float32, (None, num_labels))
#     keep_prob_ = tf.placeholder(tf.float32, name='keep')
#     learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
#
#     # Construct the LSTM inputs and LSTM cells
#     lstm_in = tf.transpose(X, [1, 0, 2])  # reshape into (seq_len, N, channels)
#     lstm_in = tf.reshape(lstm_in, [-1, num_channels])  # Now (seq_len*N, n_channels)
#
#     # To cells
#     lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
#
#     # Open up the tensor into a list of seq_len pieces
#     lstm_in = tf.split(lstm_in, seg_len, 0)
#
#     # Add LSTM layers
#     lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
#     drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
#     cell = tf.nn.rnn_cell.MultiRNNCell([drop] * lstm_layers)
#     initial_state = cell.zero_state(batch_size, tf.float32)
#
#     outputs, final_state = tf.nn.static_rnn(cell, lstm_in, dtype=tf.float32,
#                                                      initial_state=initial_state)
#
#     # We only need the last output tensor to pass into a classifier
#     y_ = tf.layers.dense(outputs[-1], num_labels)
#     print("### prediction shape: ", y_.get_shape(), " ###")
#
#     # define loss
#     # loss_math = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
#     loss_math = Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
#     print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
#     loss = -tf.reduce_mean(loss_math)
#     # print(xentropy.shape, loss.shape)
#     # define optimizer & training
#     opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     train_op = opt.minimize(loss)
#
#     # # Cost function and optimizer
#     # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
#     # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#
#     # define accuracy
#     # correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
#     correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# with graph.as_default():
#     # define
#     seg_len = 128
#     num_channels = 9
#     num_labels = 6
#     batch_size = 100
#     learning_rate = 0.001
#     num_epoches = 10000
#     training = tf.placeholder_with_default(False, shape=())
#     X = tf.placeholder(tf.float32, (None, seg_len, num_channels))
#     Y = tf.placeholder(tf.float32, (None, num_labels))
#
#     # CNN
#     # convolution layer 1
#     conv1 = tf.layers.conv1d(
#         inputs=X,
#         filters=64,
#         kernel_size=3,
#         activation=tf.nn.relu
#     )
#     print("### convolution layer 1 shape: ", conv1.shape, " ###")
#
#     # convolution layer 2
#     conv2 = tf.layers.conv1d(
#         inputs=conv1,
#         filters=64,
#         kernel_size=3,
#         activation=tf.nn.relu
#     )
#     print("### convolution layer 2 shape: ", conv2.shape, " ###")
#
#     conv2 = tf.nn.dropout(conv2, keep_prob=0.5)
#     print("### dropout layer shape: ", conv2.shape, " ###")
#
#     # pooling layer
#     pool= tf.layers.max_pooling1d(
#         inputs=conv2,
#         pool_size=4,
#         strides=2,
#         padding='same'
#     )
#     print("### pooling layer shape: ", pool.shape, " ###")
#
#     # flat
#     l_op = pool
#     shape = l_op.get_shape().as_list()
#     flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
#     print("### flat shape: ", flat.shape, " ###")
#
#     # fully connected layer 1
#     fc1 = tf.layers.dense(
#         inputs=flat,
#         units=100,
#         activation=tf.nn.relu
#     )
#     print("### fully connected layer 1 shape: ", fc1.shape, " ###")
#
#
#     # fully connected layer 3
#     fc2 = tf.layers.dense(
#         inputs=fc1,
#         units=num_labels,
#         activation=tf.nn.softmax
#     )
#     print("### fully connected layer 2 shape: ", fc2.shape, " ###")
#
#     # prediction
#     # y_ = tf.layers.batch_normalization(fc3, training=training)
#     y_ = fc2
#     print("### prediction shape: ", y_.get_shape(), " ###")
#
#     # define loss
#     # loss_math = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
#     loss_math = Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
#     print("Y shape: ", Y.shape, "y_ shape:", y_.shape)
#     loss = -tf.reduce_mean(loss_math)
#     # print(xentropy.shape, loss.shape)
#     # define optimizer & training
#     opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     train_op = opt.minimize(loss)
#
#     # # Cost function and optimizer
#     # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
#     # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#
#     # define accuracy
#     # correct = tf.nn.in_top_k(predictions=y_, targets=Y, k=1)
#     correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# # session
# with tf.Session(graph=graph) as sess:
#     tf.global_variables_initializer().run()
#     for epoch in range(num_epoches):
#         # Initialize
#         state = sess.run(initial_state)
#
#         step = 1
#         train_len = len(X_train)
#         while step * cfg.batch_size <= train_len:
#             batch_xs = util.extract_batch_size(X_train, step, cfg.batch_size)
#             # print('batch_xs', batch_xs.shape)
#             batch_ys = util.extract_batch_size(y_train, step, cfg.batch_size)
#             # print('batch_ys', batch_ys.shape)
#             features, labels = batch_xs, batch_ys
#             # _, c = sess.run([train_op, loss], feed_dict={X: batch_xs, Y: batch_ys, training: True})
#             _, c = sess.run([train_op, loss], feed_dict={X: batch_xs, Y: batch_ys, training: True, keep_prob_ : 0.5, initial_state : state, learning_rate_ : learning_rate})
#             step += 1
#         print("### Epoch: ", epoch + 1, "|Train loss = ", c,
#               "|Train acc = ", sess.run(accuracy, feed_dict={X: X_train, Y: y_train}), " ###")
#         if (epoch + 1) % 10 == 0:
#             print("### After Epoch: ", epoch + 1,
#                   " |Test acc = ", sess.run(accuracy, feed_dict={X: X_test, Y: y_test}), " ###")
#             pred_y = sess.run(tf.argmax(y_, 1), feed_dict={X: X_test})
#             cm = metrics.confusion_matrix(np.argmax(y_test, 1), pred_y, )
#             print(cm, '\n')
