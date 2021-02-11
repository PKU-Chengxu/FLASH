import tensorflow as tf
import numpy as np
import os
import json
from keras import models
from keras.layers import Dense, Activation


def regression_lookup_table(device, layer):
    with open(os.path.join('regression', device, '{}.json'.format(layer)), 'r', encoding='utf-8') as f:
        d = json.load(f)
    features = np.array(d['X'])
    labels = np.array(d['Y'])
    rate = int(0.85 * len(labels))
    train_features = features[:rate]
    test_features = features[rate:]
    train_labels = labels[:rate]
    test_labels = labels[rate:]

    model = models.Sequential([
        Dense(10),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x=train_features, y=train_labels, epochs=1000, verbose=2)
    save_dir = os.path.join('model/lookup_table', device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, '{}.h5'.format(layer)))


if __name__ == '__main__':
    device_list = ['sumsung_note10', 'redmi_note8', 'nexus6']
    layer_list = ['embedding', 'lstm', 'output']
    for d in device_list:
        for l in layer_list:
            print(d, l)
            regression_lookup_table(d, l)