from tensorflow import keras
import torch
import config as cfg

# data
n_inputs = 9
n_timesteps = 128
n_classes = 6
# lstm hidden
hidden_size = 32
n_layers = 2
batch_size=32

class CNN_Keras(keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                                         input_shape=(n_timesteps, n_inputs))
        self.conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.drop = keras.layers.Dropout(rate=0.5)
        self.maxpool = keras.layers.MaxPooling1D(pool_size=2)
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.dense2 = keras.layers.Dense(n_classes, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


class LSTM_Keras(keras.Model):

    def __init__(self):
        super().__init__()
        self.lstm1 = keras.layers.LSTM(100, input_shape=(n_timesteps, n_inputs))
        self.drop = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.dense2 = keras.layers.Dense(n_classes, activation='softmax')

    def call(self, input):
        x = self.lstm1(input)
        x = self.drop(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


class CNN_Pytorch(torch.nn.Module):

    def __init__(self):
        super(CNN_Pytorch, self).__init__();
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
        # Pytorch is channel last, but Keras is channel first
        input = input.permute(0, 2, 1)
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.max_pool1(x)
        x = self.flat(x)
        x = self.dnn1(x)
        output = self.dnn2(x)
        return output


class LSTM_Pytorch(torch.nn.Module):
    def __init__(self):
        super(LSTM_Pytorch, self).__init__()

        self.lstm1 = torch.nn.LSTM(input_size=n_inputs, hidden_size=hidden_size, num_layers=n_layers, dropout=0.5)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, dropout=0.5)
        self.fc = torch.nn.Linear(32, 6)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(0):
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, 32)
        out = self.fc(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(n_layers, cfg.batch_size, hidden_size).zero_(),
                  weight.new(n_layers, cfg.batch_size, hidden_size).zero_())
        return hidden
