batch_size = 32
n_epochs = 50

CNN_Keras = {
    'name': 'CNN_Keras',
}

LSTM_Keras = {
    'name': 'LSTM_Keras',
}

CNN_Pytorch = {
    'name': 'CNN_Pytorch',
}

LSTM_Pytorch = {
    'name': 'LSTM_Pytorch',
}

CNN_TensorFlow = {
    'name': 'CNN_TensorFlow',
}

LSTM_TensorFlow = {
    'name': 'LSTM_TensorFlow',
}

Architecture = {
    'CNN_Keras': CNN_Keras,
    'LSTM_Keras': LSTM_Keras,
    'CNN_Pytorch': CNN_Pytorch,
    'LSTM_Pytorch': LSTM_Pytorch,
    'CNN_TensorFlow': CNN_TensorFlow,
    'LSTM_TensorFlow': LSTM_TensorFlow
}

# Choose what architecure you want here:
arch = Architecture['LSTM_TensorFlow']
