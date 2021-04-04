import numpy as np


def extract_batch_size(data, step, batch_size):
    shape = list(data.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(data)
        batch[i] = data[index]

    return batch
