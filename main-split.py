# all the data from train data set, split
import numpy as np
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import to_categorical
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
    testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    # print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
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
    X = trainX
    Y = trainy
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # split into 67% for train and 33% for test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
    # fit network
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
    # save model
    model.save('./models/lenet5_weight.h5')
    # load model
    model = load_model('./models/lenet5_weight.h5')
    # evaluate model
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=verbose)
    y_predict = (y_predict > 0.01).astype(int)
    y_true = np.reshape(y_test, [-1])
    y_pred = np.reshape(y_predict, [-1])
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    return [accuracy, precision, recall, f1score]


# summarize scores
def summarize_results(scores):
    print(scores)
    mean, std = np.mean(scores), np.std(scores)
    return [mean, std]


# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    evaluate_model(trainX, trainy, testX, testy)
    # evaluation metrics
    accuracylist = list()
    precisionlist = list()
    recalllist = list()
    f1scorelist = list()
    # repeat experiment
    for n in range(repeats):
        accuracy, precision, recall, f1score = evaluate_model(trainX, trainy, testX, testy)
        print('>#%d Accuracy: %.3f' % (n + 1, accuracy * 100))
        accuracylist.append(accuracy)
        print('>#%d Precision: %.3f' % (n + 1, precision * 100))
        precisionlist.append(precision)
        print('>#%d Recall: %.3f' % (n + 1, recall * 100))
        recalllist.append(recall)
        print('>#%d F1 Score: %.3f' % (n + 1, f1score * 100))
        f1scorelist.append(f1score)
    # summarize results
    accuracy_mean, accuracy_std = summarize_results(accuracylist)
    print('Accuracy: %.3f%% (+/-%.3f)' % (accuracy_mean, accuracy_std))
    precision_mean, precision_std = summarize_results(precisionlist)
    print('Precision: %.3f%% (+/-%.3f)' % (precision_mean, precision_std))
    recall_mean, recall_std = summarize_results(recalllist)
    print('Recall: %.3f%% (+/-%.3f)' % (recall_mean, recall_std))
    f1score_mean, f1score_std = summarize_results(f1scorelist)
    print('F1 Score: %.3f%% (+/-%.3f)' % (f1score_mean, f1score_std))


run_experiment()
