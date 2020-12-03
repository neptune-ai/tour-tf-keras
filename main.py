import hashlib

import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from neptunecontrib.api import log_table
from neptunecontrib.monitoring.keras import NeptuneMonitor
from scikitplot.metrics import plot_roc, plot_precision_recall

# Select project
neptune.init('neptune-ai/tour-with-tf-keras')

# Prepare params
parameters = {'dense_units': 32,
              'activation': 'relu',
              'dropout': 0.3,
              'learning_rate': 0.05,
              'batch_size': 32,
              'n_epochs': 30}

# Create experiment
neptune.create_experiment(name='keras-training',
                          tags=['keras', 'fashion-mnist'],
                          upload_source_files=['main.py'],
                          params=parameters)

# Prepare dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Log data version
neptune.set_property('x_train_version', hashlib.md5(x_train).hexdigest())
neptune.set_property('y_train_version', hashlib.md5(y_train).hexdigest())
neptune.set_property('x_test_version', hashlib.md5(x_test).hexdigest())
neptune.set_property('y_test_version', hashlib.md5(y_test).hexdigest())

neptune.set_property('class_names', class_names)

# Log example images
for j, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 10))
    label_ = np.where(y_train == j)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[label_[0][i]], cmap=plt.cm.binary)
        plt.xlabel(class_names[j])
    neptune.log_image('train_data_sample', plt.gcf())
    plt.close('all')

# Prepare model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
    tf.keras.layers.Dropout(parameters['dropout']),
    tf.keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
    tf.keras.layers.Dropout(parameters['dropout']),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.SGD(learning_rate=parameters['learning_rate'])
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Log model summary
model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

# Train model
model.fit(x_train, y_train,
          batch_size=parameters['batch_size'],
          epochs=parameters['n_epochs'],
          callbacks=[NeptuneMonitor()])

# Log model weights
model.save('model')
neptune.log_artifact('model')

# Evaluate model
eval_metrics = model.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    neptune.log_metric('test_{}'.format(model.metrics_names[j]), metric)

# Log predictions as table
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_pred = y_pred
df = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred, 'y_pred_probability': y_pred_proba.max(axis=1)})
log_table('predictions', df)

# Log model performance visualizations
fig, ax = plt.subplots()
plot_roc(y_test, y_pred_proba, ax=ax)
neptune.log_image('model-performance-visualizations', fig, image_name='ROC')

fig, ax = plt.subplots()
plot_precision_recall(y_test, y_pred_proba, ax=ax)
neptune.log_image('model-performance-visualizations', fig, image_name='precision recall')
plt.close('all')
