batch_size = 32
n_epochs = 20

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

Architecture = {
    'CNN_Keras': CNN_Keras,
    'LSTM_Keras': LSTM_Keras,
    'CNN_Pytorch': CNN_Pytorch,
    'LSTM_Pytorch': LSTM_Pytorch
}

# Choose what architecure you want here:
arch = Architecture['LSTM_Pytorch']
