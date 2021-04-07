# input library
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import netron
import torch.onnx
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.autograd import Variable

# py library
import util
import data_file as df
import data_load as dl
import model as model
import config as cfg

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
# print('y_train shape', y_train.shape)
y_test = dl.load_y(y_test_path)
# print('y_test shape', y_test.shape)


if __name__ == '__main__':

    # find model and backend
    model_name = cfg.arch['name']
    type = model_name.split('_')[0]
    backend = model_name.split('_')[1]
    if model_name == 'CNN_Keras':
        net = model.CNN_Keras().getModel()
    elif model_name == 'LSTM_Keras':
        net = model.LSTM_Keras().getModel()
    elif model_name == 'CNN_Pytorch':
        net = model.CNN_Pytorch()
    elif model_name == 'LSTM_Pytorch':
        net = model.LSTM_Pytorch()
    elif model_name == 'CNN_TensorFlow':
        net = model.CNN_TensorFlow()
    elif model_name == 'LSTM_TensorFlow':
        net = model.LSTM_TensorFlow()

    # Keras Model
    if backend == 'Keras':
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        y_train = to_categorical(y_train)
        net.fit(X_train, y_train, epochs=cfg.n_epochs, batch_size=cfg.batch_size)
        tf.keras.models.save_model(net, './models/' + model_name + '.h5')
        net = tf.python.keras.models.load_model('./models/' + model_name + '.h5')
        netron.start('./models/' + model_name + '.h5')

        # evaluation
        y_predict = net.predict(X_test, batch_size=cfg.batch_size)
        y_predict = np.argmax(y_predict, axis=1)
        y_true = np.reshape(y_test, [-1])
        y_pred = np.reshape(y_predict, [-1])
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')
        print([accuracy, precision, recall, f1score])

    # Pytorch Model
    if backend == 'Pytorch':

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # net = net.to(device)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0015)

        epochs = cfg.n_epochs
        log_step_freq = 100

        print("Start Training...")
        best_accuracy = 0.0
        best_model = None
        epoch_train_losses = []
        epoch_train_acc = []
        epoch_test_losses = []
        epoch_test_acc = []
        params = {
            'best_model': best_model,
            'epochs': [],
            'train_loss': [],
            'test_loss': [],
            'lr': [],
            'train_accuracy': [],
            'test_accuracy': []
        }

        for epoch in range(1, epochs + 1):
            net.train()
            loss_sum = 0.0
            metric_sum = 0.0
            step = 1
            train_losses = []
            train_accuracy = 0
            if type == 'LSTM':
                h = net.init_hidden()

            train_len = len(X_train)
            while step * cfg.batch_size <= train_len:
                batch_xs = util.extract_batch_size(X_train, step, cfg.batch_size)
                batch_ys = util.extract_batch_size(y_train, step, cfg.batch_size)
                features, labels = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
                optimizer.zero_grad()
                predictions = net(features.float())
                loss = loss_func(predictions, labels.long())
                train_losses.append(loss.item())
                top_p, top_class = predictions.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape).long()
                train_accuracy += torch.mean(equals.type(torch.FloatTensor))
                equals = top_class
                loss.backward()
                optimizer.step()
                step += 1

            p = optimizer.param_groups[0]['lr']
            params['lr'].append(p)
            params['epochs'].append(epoch)
            train_loss_avg = np.mean(train_losses)
            train_accuracy_avg = train_accuracy / (step - 1)
            epoch_train_losses.append(train_loss_avg)
            epoch_train_acc.append(train_accuracy_avg)
            print("Epoch: {}/{}...".format(epoch, epochs),
                  ' ' * 16 + "Train Loss: {:.4f}".format(train_loss_avg),
                  "Train accuracy: {:.4f}...".format(train_accuracy_avg))

        print('Finished Training...')
        torch.save(net, './models/' + model_name + '.pkl')
        net = torch.load('./models/' + model_name + '.pkl')
        x = Variable(torch.FloatTensor(32, 128, 9))
        torch.onnx.export(net, x, './models/' + model_name + '.onnx')
        netron.start('./models/' + model_name + '.onnx')

        print('Start Testing...')
        net.eval()
        test_losses = []
        test_accuracy = 0
        test_precision = 0
        test_recall = 0
        test_f1score = 0
        step = 1
        if type == 'LSTM':
            test_h = net.init_hidden()

        test_len = len(X_test)
        while step * cfg.batch_size <= test_len:
            batch_xs = util.extract_batch_size(X_test, step, cfg.batch_size)
            batch_ys = util.extract_batch_size(y_test, step, cfg.batch_size)

            features, labels = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))

            predictions = net(features.float())
            test_loss = loss_func(predictions, labels.long())
            test_losses.append(test_loss.item())

            top_p, top_class = predictions.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape).long()

            test_accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_precision += precision_score(top_class.cpu(), labels.view(*top_class.shape).long().cpu(),
                                              average='macro')
            test_recall += recall_score(top_class.cpu(), labels.view(*top_class.shape).long().cpu(), average='macro')
            test_f1score += f1_score(top_class.cpu(), labels.view(*top_class.shape).long().cpu(), average='macro')
            step += 1

        test_loss_avg = np.mean(test_losses)
        test_f1_avg = test_f1score / (step - 1)
        test_precision_avg = test_precision / (step - 1)
        test_recall_avg = test_recall / (step - 1)
        test_accuracy_avg = test_accuracy / (step - 1)
        if (test_accuracy_avg > best_accuracy):
            best_accuracy = test_accuracy_avg
            best_model = copy.deepcopy(net)

        print([test_loss_avg, test_accuracy_avg, test_precision_avg, test_recall_avg, test_f1_avg, best_accuracy,
               best_model])
        print('Finished Testing...')

    # TensorFlow Model
    if backend == 'TensorFlow':

        n_steps = len(X_train[0])  # 128 timesteps per series
        n_input = len(X_train[0][0])  # 9 input parameters per timestep
        n_classes = 6  # Total 6 classes

        # Training
        learning_rate = 0.0025
        lambda_loss_amount = 0.0015

        # Graph input/output
        x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        pred = net.getGraph(x)
        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )  # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Launch the graph
        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        epochs = cfg.n_epochs
        for epoch in range(1, epochs + 1):
            train_losses = []
            train_accuracies = []
            step = 1

            train_len = len(X_train)
            while step * cfg.batch_size <= train_len:
                batch_xs = util.extract_batch_size(X_train, step, cfg.batch_size)
                batch_ys = util.extract_batch_size(y_train, step, cfg.batch_size)

                # Fit training using batch data
                _, loss, acc = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={
                        x: batch_xs,
                        y: util.one_hot(batch_ys, n_classes)
                    }
                )
                train_losses.append(loss)
                train_accuracies.append(acc)

                # print("Training iter #" + str(step * batch_size) + \
                #       ":   Batch Loss = " + "{:.6f}".format(loss) + \
                #       ", Accuracy = {}".format(acc))

                step += 1

            print("Epoch: {}/{}...".format(epoch, epochs) + \
                  ":   Epoch Loss = " + "{:.6f}".format(np.mean(train_losses)) + \
                  ", Accuracy = {}".format(np.mean(train_accuracies)))

        # test data
        one_hot_predictions, test_accuracy, test_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={
                x: X_test,
                y: util.one_hot(y_test, n_classes)
            }
        )

        print("FINAL RESULT: " + \
              "Test Loss = {}".format(test_loss) + \
              ", Accuracy = {}".format(test_accuracy))
