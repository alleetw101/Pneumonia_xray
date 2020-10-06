import pathlib
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Import data and count files
train_data_dir = pathlib.Path('train')
test_data_dir = pathlib.Path('test')
train_image_count = len(list(train_data_dir.glob('*/*.jpeg')))
test_image_count = len(list(test_data_dir.glob('*/*.jpeg')))

# Loads datasets into tensorflow dataset
list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=False)
list_ds = list_ds.shuffle(train_image_count, reshuffle_each_iteration=False)
test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)

# Obtaining categorical names with folder names
class_names = np.array(sorted([item.name for item in train_data_dir.glob('*') if item.name != '.DS_Store']))

# Creating training and validation dataset from training dataset
val_size = int(train_image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)


# Functions to create image, label pair in dataset
def get_label(file_path):
    parts = tf.strings.split(file_path, sep='/')
    temp = parts[-2] == class_names
    return tf.argmax(temp)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=1)
    return tf.image.resize_with_pad(img, 540, 720)


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Applies image/label functions to train/val/test datasets to create model datasets
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# Optimize dataset performance for model
def configure_performance(ds):
    ds = ds.shuffle(1000)
    ds = ds.batch(64)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = configure_performance(train_ds).repeat()
val_ds = configure_performance(val_ds)
test_ds = configure_performance(test_ds)

# Model parameters
ACTIVATION = 'relu'
REGULARIZER = tf.keras.regularizers.l2(0.01)

# Model creation and compile
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.Dropout(0.2),
    layers.Dense(512, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.Dropout(0.2),
    layers.Dense(128, activation=ACTIVATION, kernel_regularizer=REGULARIZER),
    layers.Dense(2)
])
# testing
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Model dataset training
model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=1, steps_per_epoch=16,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)])

# Model evaluation of test dataset
model.evaluate(test_ds, verbose=1)
