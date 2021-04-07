# TensorFlow library
import tensorflow.compat.v1 as tf
# Keras library
from tensorflow import keras
# Pytorch library
import torch
from torchsummary import summary
# Config
import config as cfg

# data
# Channel: TensorFlow and Pytorch is channel last, but Keras is channel first
n_inputs = 9
n_timesteps = 128
n_classes = 6
# lstm
hidden_size = 32
n_layers = 1


# Use Functional model because Sequential model is special Functional model
class CNN_Keras():

    def getModel(self):
        inputs = keras.Input(shape=(n_timesteps, n_inputs))
        x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=100, activation='relu')(x)
        outputs = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_Keras_Model')
        # model.summary()
        return model


# Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model
# However, this subclass model can show Keras has the same forward function as Tensorflow and Pytorch
# class CNN_Keras(keras.Model):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_inputs))
#         self.conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')
#         self.drop = keras.layers.Dropout(rate=0.5)
#         self.maxpool = keras.layers.MaxPooling1D(pool_size=2)
#         self.flat = keras.layers.Flatten()
#         self.dense1 = keras.layers.Dense(units=100, activation='relu')
#         self.dense2 = keras.layers.Dense(n_classes, activation='softmax')
#
#     def call(self, input):
#         x = self.conv1(input)
#         x = self.conv2(x)
#         x = self.drop(x)
#         x = self.maxpool(x)
#         x = self.flat(x)
#         x = self.dense1(x)
#         output = self.dense2(x)
#         return output


class CNN_Pytorch(torch.nn.Module):

    def __init__(self):
        super(CNN_Pytorch, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=n_inputs, out_channels=64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.drop = torch.nn.Dropout(p=0.5)
        self.max_pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.flat = torch.nn.Flatten()
        self.dnn1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=int((n_timesteps - 4) / 2 * 64), out_features=100),
            torch.nn.ReLU()
        )
        self.dnn2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=100, out_features=n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        input = input.permute(0, 2, 1)
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.max_pool1(x)
        x = self.flat(x)
        x = self.dnn1(x)
        output = self.dnn2(x)
        return output


class CNN_TensorFlow():

    def getGraph(self, x):
        # convolution layer 1
        x = tf.layers.conv1d(inputs=x, filters=64, kernel_size=3, activation=tf.nn.relu)
        print("### convolution layer 1 shape: ", x.shape, " ###")
        # convolution layer 2
        x = tf.layers.conv1d(inputs=x, filters=64, kernel_size=3, activation=tf.nn.relu)
        print("### convolution layer 2 shape: ", x.shape, " ###")
        # dropout layer
        x = tf.nn.dropout(x, keep_prob=0.5)
        print("### dropout layer shape: ", x.shape, " ###")
        # pooling layer
        x = tf.layers.max_pooling1d(inputs=x, pool_size=2, strides=2)
        print("### pooling layer shape: ", x.shape, " ###")
        # flatten layer
        x = tf.layers.flatten(inputs=x)
        print("### flat shape: ", x.shape, " ###")
        # fully connected layer 1
        x = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu)
        print("### fully connected layer 1 shape: ", x.shape, " ###")
        # fully connected layer 3
        x = tf.layers.dense(inputs=x, units=n_classes, activation=tf.nn.softmax)
        print("### fully connected layer 2 shape: ", x.shape, " ###")
        return x


class LSTM_Keras():

    def getModel(self):
        inputs = keras.Input(shape=(n_timesteps, n_inputs))
        x = keras.layers.LSTM(units=32)(inputs)
        x = keras.layers.Dropout(rate=0.5)(x)
        outputs = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Keras_Model')
        # model.summary()
        return model


class LSTM_Pytorch(torch.nn.Module):

    def __init__(self):
        super(LSTM_Pytorch, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=n_inputs, hidden_size=hidden_size)
        # self.lstm = torch.nn.LSTM(input_size=32, hidden_size=32, num_layers=2, dropout=0.5)
        self.drop = torch.nn.Dropout(p=0.5)
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        input = input.permute(1, 0, 2)
        x, hidden = self.lstm(input, self.hidden)
        x = self.drop(x)
        x = x[-1]
        x = x.contiguous().view(-1, 32)
        output = self.dnn(x)
        return output

    def init_hidden(self):
        weight = next(self.parameters()).data
        self.hidden = (weight.new(n_layers, cfg.batch_size, hidden_size).zero_(),
                       weight.new(n_layers, cfg.batch_size, hidden_size).zero_())


class LSTM_TensorFlow():

    def getGraph(self, x):
        _weights = {
            'hidden': tf.Variable(tf.random_normal([n_inputs, hidden_size])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([hidden_size, n_classes], mean=1.0))
        }
        _biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_size])),  # Hidden layer bias
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        x = tf.transpose(x, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        x = tf.reshape(x, [-1, n_inputs])  # new shape: (n_steps*batch_size, n_input)

        # ReLU activation
        x = tf.nn.relu(tf.matmul(x, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        x = tf.split(x, n_timesteps, 0)  # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.nn.static_rnn(lstm_cells, x, dtype=tf.float32)

        # Get last time step's output feature for a "many-to-one" style classifier
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


if __name__ == '__main__':
    # Keras Model
    LSTM_Keras().getModel().summary()
    # Pytorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Pytorch().to(device)
    summary(model, (128, 9))
