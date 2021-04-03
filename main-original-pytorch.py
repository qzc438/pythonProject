# train from train data set (75%), test from test data set (25%)
import numpy as np
import onnxmltools as onnxmltools
import datetime
import pandas as pd
from pandas import read_csv
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import torch.nn.functional as F

import torch
import torch.nn as nn


# def load_file(filepath):
#     dataframe = read_csv(filepath, header=None, delim_whitespace=True)
#     return dataframe.values
#
#
# # load a list of files into a 3D array of [samples, timesteps, features]
# def load_group(filenames, prefix=''):
#     loaded = list()
#     for name in filenames:
#         data = load_file(prefix + name)
#         loaded.append(data)
#     # stack group so that features are the 3rd dimension
#     loaded = np.dstack(loaded)
#     return loaded
#
#
# # load a dataset group, such as train or test
# def load_dataset_group(group, prefix=''):
#     filepath = prefix + group + '/Inertial Signals/'
#     # load all 9 files as a single array
#     filenames = list()
#     # total acceleration
#     filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
#     # body acceleration
#     filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
#     # body gyroscope
#     filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
#     # load input data
#     X = load_group(filenames, filepath)
#     # load class output
#     y = load_file(prefix + group + '/y_' + group + '.txt')
#     return X, y
#
#
# # load the dataset, returns train and test X and y elements
# def load_dataset(prefix=''):
#     # load all train
#     trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
#     # print(trainX.shape, trainy.shape)
#     # load all test
#     testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
#     # print(testX.shape, testy.shape)
#     # zero-offset class values， begin with 0, not 1
#     trainy = trainy - 1
#     testy = testy - 1
#     # one hot encode y
#     # trainy = to_categorical(trainy)
#     # testy = to_categorical(testy)
#     print(trainX.shape, trainy.shape, testX.shape, testy.shape)
#     return trainX, trainy, testX, testy

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

TRAIN = "train/"
TEST = "test/"
DATASET_PATH = "./UCI HAR Dataset/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch[i] = _train[index]

    return batch

trainX = load_X(X_train_signals_paths)
testX = load_X(X_test_signals_paths)
trainy= load_y(y_train_path)
testy = load_y(y_test_path)


class LSTM(nn.Module):
    def __init__(self, ):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(9, 32, 2, dropout=0.5)
        self.lstm2 = nn.LSTM(9, 32, 2, dropout=0.5)
        self.fc = nn.Linear(32, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(0):
            # x = F.relu(x)
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, 32)
        out = self.fc(out)
        out = F.softmax(out,dim=1)

        return out

    def init_hidden(self):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        # if (torch.cuda.is_available()):
        #     hidden = (weight.new(2, 32, 32).zero_().cuda(),
        #               weight.new(2, 32, 32).zero_().cuda())
        # else:
        #     hidden = (weight.new(2, 32, 32).zero_(),
        #               weight.new(2, 32, 32).zero_())
        hidden = (weight.new(2, 32, 32).zero_(),
                      weight.new(2, 32, 32).zero_())
        return hidden

# class CNN(nn.Module):
#     def __init__(self, n_timesteps, n_features, n_outputs):
#         super(CNN, self).__init__();
#         self.layers = nn.ModuleList([
#             nn.Conv1d(in_channels=n_timesteps, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(in_features=32*n_timesteps, out_features=100),
#             nn.ReLU(),
#             nn.Linear(in_features=100, out_features=n_outputs),
#             nn.Softmax(dim=1)
#         ])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

class CNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN, self).__init__();
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.drop = nn.Dropout(0.5)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        self.flat = nn.Flatten()
        # self.dnn1 = nn.Linear(in_features=32 * n_timesteps, out_features=100)
        # self.dnn2 = nn.Linear(in_features=100, out_features=n_outputs)
        self.dnn1 = nn.Linear(in_features=3968, out_features=100)
        self.dnn2 = nn.Linear(in_features=100, out_features=6)

        # nn.Dropout()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print('x.shape------------------', x.shape)
        x = F.relu(self.conv1(x))
        # print('x.shape------------------', x.shape)
        x = F.relu(self.conv2(x))
        # print('x.shape------------------', x.shape)
        x = self.drop(x)
        # print('x.shape------------------', x.shape)
        x = self.max_pool1(x)
        # x = x.view(-1, 32*n_timesteps)
        # print('x.shape------------------', x.shape)
        x = self.flat(x)
        # print('x.shape------------------', x.shape)
        x = F.relu(self.dnn1(x))
        # print('x.shape------------------', x.shape)
        x = F.softmax(self.dnn2(x), dim=1)
        return x


def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)


if __name__ == '__main__':
    # trainX, trainy, testX, testy = load_dataset()
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = CNN(n_timesteps, n_features, n_outputs)
    # model = LSTM()

    # in_dim = 1152
    # hidden_dim = 128
    # out_dim = 6
    # model = nn.Sequential(
    #     nn.Linear(in_dim, hidden_dim),
    #     nn.ReLU(),
    #     nn.Linear(hidden_dim, out_dim),
    #     nn.Softmax(dim = 1)
    #     )
    # summary(model, input_shape=(128, 9))
    # import pdb; pdb.set_trace()
    # print('trainX shape', trainX.shape)
    # # trainX = np.transpose(trainX, (0, 2, 1))
    # print('trainX shape', trainX.shape)
    # trainX = torch.utils.data.DataLoader(trainX, batch_size=32, shuffle=True, num_workers=0)
    # trainy = torch.utils.data.DataLoader(trainy, batch_size=32, shuffle=True, num_workers=0)
    # # testX = np.transpose(testX, (0, 2, 1))
    # testX = torch.utils.data.DataLoader(testX, batch_size=32, shuffle=True, num_workers=0)
    # testy = torch.utils.data.DataLoader(testy, batch_size=32, shuffle=True, num_workers=0)

    # for i, x in enumerate(trainX):
    #     # print('x', x)
    #     out = model(Variable(x.float()))
    #     print('out', out)

    def accuracy(y_pred, y_true):
        y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
        return accuracy_score(y_true, y_pred_cls)


    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0015)
    metric_func = accuracy
    metric_name = "accuracy"

    epochs = 100
    log_step_freq = 100

    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)


    best_accuracy = 0.0
    best_model = None
    epoch_train_losses = []
    epoch_train_acc = []
    epoch_test_losses = []
    epoch_test_acc = []
    params = {
        'best_model' : best_model,
        'epochs' : [],
        'train_loss' : [],
        'test_loss' : [],
        'lr' : [],
        'train_accuracy' : [],
        'test_accuracy' : []
    }

    for epoch in range(1, epochs + 1):

        # 1，训练循环-------------------------------------------------
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        # h = model.init_hidden()
        train_losses = []
        train_accuracy = 0

        # stop_index = len(trainy)//32
        # start_index = 0
        # for features, labels in zip(trainX, trainy):
        train_len = len(trainX)
        while step * 32 <= train_len:
            batch_xs = extract_batch_size(trainX, step, 32)
            # batch_ys = one_hot_vector(extract_batch_size(y_train, step, batch_size))
            batch_ys = extract_batch_size(trainy, step, 32)

            features, labels = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
            # start_index += 1
            # if start_index >= stop_index -1:
            #     break
            # print('step', step)
            # print('features shape', features.shape)
            # print('features0', features[0][0])
            # print('features1', features[1][0])

            # features = torch.reshape(features,(32, -1))
            # print('labels', labels)

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
            # h = tuple([each.data for each in h])
            # 梯度清零
            optimizer.zero_grad()
            predictions = model(features.float())
            # predictions = model(features.float(), h)
            # print('predictions shape:', predictions.shape)
            # one_hot_pred = torch.zeros(size = (32, 6))
            # for i in predictions:
            #     one_hot_pred[i] = predictions[i]
            # predictions = to_categorical(predictions)
            # print('predictions', predictions)
            # predictions = torch.argmax(predictions,dim=1)
            # _, predictions = torch.max(predictions.data,dim=1)
            # predictions = np.squeeze(predictions)
            # print('predictions', predictions)
            # labels = torch.argmax(labels, dim=1)
            # print('labels', labels)
            # print('labels shape:', labels.shape)
            # loss = loss_func(predictions, labels.squeeze(1))
            loss = loss_func(predictions, labels.long())
            # # print('loss', loss)
            # metric = metric_func(predictions, labels)
            #
            #
            # # 反向传播求梯度
            # loss.backward()
            # optimizer.step()

            # # 打印batch级别日志
            # loss_sum += loss.item()
            # metric_sum += metric.item()
            # if step % log_step_freq == 0:
            #     print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
            #           (step, loss_sum / step, metric_sum / step))
            # step = step + 1
            train_losses.append(loss.item())

            top_p, top_class = predictions.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape).long()
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))
            equals = top_class

            loss.backward()
            # clip_grad.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            step += 1

        p = optimizer.param_groups[0]['lr']
        params['lr'].append(p)
        params['epochs'].append(epoch)
        # sched.step()
        train_loss_avg = np.mean(train_losses)
        train_accuracy_avg = train_accuracy / (step - 1)
        epoch_train_losses.append(train_loss_avg)
        epoch_train_acc.append(train_accuracy_avg)
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              ' ' * 16 + "Train Loss: {:.4f}".format(train_loss_avg),
              "Train accuracy: {:.4f}...".format(train_accuracy_avg))

        # # 2，验证循环-------------------------------------------------
        # model.eval()
        # val_loss_sum = 0.0
        # val_metric_sum = 0.0
        # val_step = 1
        #
        # # stop_index = len(testy)//32
        # # start_index = 0
        # with torch.no_grad():
        #     correct = 0
        #     total = 0
        # for features, labels in zip(testX, testy):
        #     # start_index += 1
        #     # if start_index > stop_index:
        #     #     break
        #     predictions = model(features.float())
        #     # predictions = torch.argmax(predictions, dim=1)
        #     _, pre = torch.max(predictions.data, 1)
        #     # labels = torch.argmax(labels, dim=1)
        #     val_loss = loss_func(predictions, labels.squeeze(1))
        #     val_metric = metric_func(predictions, labels)
        #     # print('val_loss', val_loss)
        #     total += labels.size(0)
        #     correct += (pre == labels).sum().item()
        # print('Accuracy: {}'.format(correct / total))
        #
        # val_loss_sum += val_loss.item()
        # val_metric_sum += val_metric.item()
        # val_step = val_step + 1
        #
        # # 3，记录日志-------------------------------------------------
        # info = (epoch, loss_sum / step, metric_sum / step,
        #         val_loss_sum / val_step, val_metric_sum / val_step)
        # dfhistory.loc[epoch - 1] = info
        #
        # # 打印epoch级别日志
        # print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
        #        "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
        #       % info)
        # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')
    # dummy_input = torch.randn(32, 128, 9, requires_grad=True)
    # torch.onnx.export(model, dummy_input, "cnn-pytorch.onnx")
