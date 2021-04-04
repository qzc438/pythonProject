from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
from pandas import read_csv
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

app = Flask(__name__)


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
    # trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
    # print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'UserTempData/')
    # print(testX.shape, testy.shape)
    # zero-offset class values
    # trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    # trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(testX.shape, testy.shape)
    return testX, testy


# summarize scores
def summarize_results(scores):
    print('scores:', scores)
    mean, std = np.mean(scores), np.std(scores)
    return [mean, std]


# run an experiment
def run_experiment(model_name):
    # load data
    testX, testy = load_dataset()
    # load model
    testy = np.argmax(testy, axis=1)
    y_true = np.reshape(testy, [-1])
    # evaluation metrics
    sess = onnxruntime.InferenceSession('./models/' + model_name + '.onnx')
    y_predict = sess.run(None, {sess.get_inputs()[0].name: testX.astype(np.float32)})
    y_predict = np.array(y_predict)
    y_predict = np.argmax(y_predict, axis=2)
    y_pred = np.reshape(y_predict, [-1])
    print('y_predict', y_predict)
    print('testy', testy)
    # evaluation
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')
    print('>Accuracy: %.2f%%' % (accuracy * 100))
    print('>Precision: %.2f%%' % (precision * 100))
    print('>Recall: %.2f%%' % (recall * 100))
    print('>F1 Score: %.2f%%' % (f1score * 100))
    return [accuracy, precision, recall, f1score]


# compile model
@app.route('/compile', methods=['post','get'])
def compile():
    model_name = request.args.get('model_name')
    print('model_name: ', model_name)
    accuracy, precision, recall, f1score = run_experiment(model_name)
    data = {'Accuracy': '%.2f%%' % (accuracy * 100), 'Precision': '%.2f%%' % (precision * 100), 'Recall': '%.2f%%' % (recall * 100), 'F1 Score': '%.2f%%' % (f1score * 100)}
    return jsonify(data)


# others
@app.route('/')
def login():
    return 'server start successfully'


# start server
if __name__ == '__main__':
    app.run()
