# train from train data set (75%), test from test data set (25%)
import numpy as np
import onnxmltools as onnxmltools
import datetime
import pandas as pd
from pandas import read_csv
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import torch
import torch.nn as nn


# load a single file as a numpy array
from tensorflow.python.keras.utils.vis_utils import plot_model
from torch.autograd import Variable


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
    # print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    # print(testX.shape, testy.shape)
    # zero-offset class values， begin with 0, not 1
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


class CNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN, self).__init__();
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=int(64 * (n_timesteps - 4) / 2), out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=n_outputs),
            nn.Softmax(dim=1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)


if __name__ == '__main__':
    trainX, trainy, testX, testy = load_dataset()
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    in_dim = 1152
    hidden_dim = 128
    out_dim = 6

    model = CNN(n_timesteps, n_features, n_outputs)
    # model = nn.Sequential(
    #     nn.Linear(in_dim, hidden_dim),
    #     nn.ReLU(),
    #     nn.Linear(hidden_dim, out_dim),
    #     nn.Softmax(dim = 1)
    #     )
    summary(model, input_shape=(9, 128))
    trainX = np.transpose(trainX, (0, 2, 1))
    trainX = torch.utils.data.DataLoader(trainX, batch_size=32, shuffle=True, num_workers=0)
    trainy = torch.utils.data.DataLoader(trainy, batch_size=32, shuffle=True, num_workers=0)
    testX = np.transpose(testX, (0, 2, 1))
    testX = torch.utils.data.DataLoader(testX, batch_size=32, shuffle=True, num_workers=0)
    testy = torch.utils.data.DataLoader(testy, batch_size=32, shuffle=True, num_workers=0)
    # for i, x in enumerate(trainX):
    #     # print('x', x)
    #     out = model(Variable(x.float()))
    #     print('out', out)

    def accuracy(y_pred, y_true):
        y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
        return accuracy_score(y_true, y_pred_cls)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    metric_func = accuracy
    metric_name = "accuracy"

    epochs = 3
    log_step_freq = 100

    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)

    for epoch in range(1, epochs + 1):

        # 1，训练循环-------------------------------------------------
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        # stop_index = len(trainy)//32
        # start_index = 0
        for features, labels in zip(trainX, trainy):
            # start_index += 1
            # if start_index >= stop_index -1:
            #     break
            # print('step', step)
            print('features shape', features.shape)
            # print('features0', features[0][0])
            # print('features1', features[1][0])

            # features = torch.reshape(features,(32, -1))
            # print('labels', labels)

            # 梯度清零
            optimizer.zero_grad()
            # print(features[0])
            # print(features[1])
            # predictions0 = model(features[0].unsqueeze(0).float())
            # predictions1 = model(features[1].unsqueeze(0).float())
            # print(f'predictions0 is {predictions0}')
            # print(f'prediction1 is {predictions1}')
            # import pdb; pdb.set_trace()
            # 正向传播求损失
            # predictions = model(features[:32].float())
            # predictions = model(features.reshape(32, -1).float())
            predictions = model(features.float())
            # print('predictions shape:', predictions.shape)
            # print('predictions', predictions)
            # predictions = torch.argmax(predictions,dim=1)
            print(predictions)
            labels = torch.argmax(labels, dim=1)
            # print('labels', labels)
            # print('labels shape:', labels.shape)
            loss = loss_func(predictions, labels)
            metric = metric_func(predictions, labels)

            # 反向传播求梯度
            loss.backward()
            optimizer.step()

            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))
            step = step + 1
        # 2，验证循环-------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        # stop_index = len(testy)//32
        # start_index = 0
        for features, labels in zip(testX, testy):
            # start_index += 1
            # if start_index > stop_index:
            #     break
            with torch.no_grad():
                predictions = model(features.float())
                # predictions = torch.argmax(predictions, dim=1)
                labels = torch.argmax(labels, dim=1)
                val_loss = loss_func(predictions, labels)
                val_metric = metric_func(predictions, labels)

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()
            val_step = val_step + 1

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
               "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
              % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')
