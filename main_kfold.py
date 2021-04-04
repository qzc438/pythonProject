# all the data from train data set, k-fold validation
import numpy as np
import onnxmltools as onnxmltools
from pandas import read_csv
from sklearn.model_selection import KFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


# load a single file as a numpy array
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
    # testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    # print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    # testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    # testy = to_categorical(testy)
    print(trainX.shape, trainy.shape)
    return trainX, trainy


# create model
def create_model(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # model structure
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit and evaluate a model
def evaluate_model(trainX, trainy):
    verbose, epochs, batch_size = 0, 10, 32
    # random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # k-fold validation
    X = trainX
    Y = trainy
    n_split = 10
    # evaluation metrics
    accuracylist = list()
    precisionlist = list()
    recalllist = list()
    f1scorelist = list()
    # repeat calculator
    number = 0
    for train_index, test_index in KFold(n_split).split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # create model
        model = create_model(trainX, trainy)
        # fit network
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        # save model
        model.save('./models/model' + str(number+1) + '.h5')
        keras_model = load_model('./models/model' + str(number+1) + '.h5')
        onnx_model = onnxmltools.convert_keras(keras_model)
        onnxmltools.utils.save_model(onnx_model, './models/model' + str(number+1) + '.onnx')
        # evaluate model
        y_predict = model.predict(x_test, batch_size=batch_size, verbose=verbose)
        print('#' + str(number+1) + '#' + 'y_predict:', y_predict)
        y_predict = np.argmax(y_predict, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_true = np.reshape(y_test, [-1])
        y_pred = np.reshape(y_predict, [-1])
        accuracy = accuracy_score(y_true, y_pred)
        accuracylist.append(accuracy)
        precision = precision_score(y_true, y_pred, average='macro')
        precisionlist.append(precision)
        recall = recall_score(y_true, y_pred, average='macro')
        recalllist.append(recall)
        f1score = f1_score(y_true, y_pred, average='macro')
        f1scorelist.append(f1score)
        number = number + 1
    # summarize results
    accuracy_mean, accuracy_std = summarize_results(accuracylist)
    print('K-fold Accuracy: %.2f%% (+/-%.2f)' % (accuracy_mean*100, accuracy_std))
    precision_mean, precision_std = summarize_results(precisionlist)
    print('K-fold Precision: %.2f%% (+/-%.2f)' % (precision_mean*100, precision_std))
    recall_mean, recall_std = summarize_results(recalllist)
    print('K-fold Recall: %.2f%% (+/-%.2f)' % (recall_mean*100, recall_std))
    f1score_mean, f1score_std = summarize_results(f1scorelist)
    print('K-fold F1 Score: %.2f%% (+/-%.2f)' % (f1score_mean*100, f1score_std))


# summarize scores
def summarize_results(scores):
    print('scores:', scores)
    mean, std = np.mean(scores), np.std(scores)
    return [mean, std]


# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy = load_dataset()
    # evaluation metrics
    evaluate_model(trainX, trainy)


run_experiment()
