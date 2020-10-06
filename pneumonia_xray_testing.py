import pathlib
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Importing data
train_data_dir = pathlib.Path('train')
test_data_dir = pathlib.Path('test')

# Verifying datasets were loaded correctly. Counts number of jpeg files in folder/directory
train_image_count = len(list(train_data_dir.glob('*/*.jpeg')))
test_image_count = len(list(test_data_dir.glob('*/*.jpeg')))
print(train_image_count, test_image_count)

# Testing Variables
img_width = 720
img_height = 540
val_size = int(train_image_count * 0.2)
test_size = int(train_image_count * 0.1)
BATCH_SIZE = 64
ACTIVATION = 'relu'

regularizer = tf.keras.regularizers.l2(0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

EPOCHS = 50  # Max Epoch w/ EarlyStopping
STEPS_PER_EPOCH = 16

# Variable names for log file
REGULARIZER = 'l2(0.01)'
OPTIMIZER = 'adam'
CALLBACK = 'EarlyStopping(val_loss, 2)'
COMMENTS = '(training) shuffle, repeat, batch'

# Obtaining categorical names with folder names
class_names = np.array(sorted([item.name for item in train_data_dir.glob('*') if item.name != '.DS_Store']))
print(class_names)


# Functions to create image, label pair in dataset
def get_label(file_path):
    parts = tf.strings.split(file_path, sep='/')
    temp = parts[-2] == class_names
    return tf.argmax(temp)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=1)
    return tf.image.resize_with_pad(img, img_height, img_width)


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Configures datasets for use in models
def configure_performance(ds, dataset=''):
    ds = ds.shuffle(1000)
    if dataset == 'train':
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


average_acc = []
average_loss = []

for _ in range(8):
    # Time variables for log file
    date = datetime.now().strftime("%Y%m%d_%H%M")
    starttime = time.time()

    # Loads datasets into tensorflow dataset with verification
    list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'), shuffle=True)
    # list_ds = list_ds.shuffle(train_image_count, reshuffle_each_iteration=False)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*'), shuffle=False)

    # for i in list_ds.take(1):
    #     print(i.numpy())
    #     example = i

    # Splitting training dataset into train and validation datasets
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    print(tf.data.experimental.cardinality(train_ds).numpy(), tf.data.experimental.cardinality(val_ds).numpy())

    # Creates image/label pair for datasets with verification
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # for image, label in train_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())

    # Configure datasets for model
    train_ds = configure_performance(train_ds, dataset='train')
    val_ds = configure_performance(val_ds)
    test_ds = configure_performance(test_ds)

    # Create and compile model
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.Dropout(0.2),
        layers.Dense(512, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.Dropout(0.2),
        layers.Dense(128, activation=ACTIVATION, kernel_regularizer=regularizer),
        layers.Dense(2)
    ])

    model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  # Update log file variables if metric is changed

    # Train model on testing dataset
    train_history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1, steps_per_epoch=STEPS_PER_EPOCH,
                              callbacks=callback)

    # Time variables for log file
    endtime = time.time()
    traintime = (endtime-starttime)

    # Evaluate trained model on test dataset
    loss, acc = model.evaluate(test_ds, verbose=1)

    # Saves entire model in SavedModel format
    model.save(f'PneSavedModels/Pne{date}', overwrite=False)

    # Update log file
    with open('PneSavedModels/PneLog.txt', 'r') as f:
        tempfile = f.read()

    with open('PneSavedModels/PneLog.txt', 'w') as f:
        f.write('###############\n')
        f.write(f'Run: {date}, train_time: {traintime}\n')
        f.write(f'Img_width: {img_width}, Img_height: {img_height}, Val_size: {val_size}\n')
        f.write(f'Batch_size: {BATCH_SIZE}, Activation: {ACTIVATION}, Optimizer: {OPTIMIZER}, Regularizer: {REGULARIZER}\n')
        f.write(f'Epochs: {EPOCHS}, Steps_per_epoch: {STEPS_PER_EPOCH}, Callback(s): {CALLBACK}\n\n')
        for runs in range(len(train_history.history['loss'])):
            f.write(f'Epoch {runs+1}- Loss: {train_history.history["loss"][runs]}, ')
            f.write(f'Acc: {train_history.history["accuracy"][runs]}, ')
            f.write(f'Val_loss: {train_history.history["val_loss"][runs]}, ')
            f.write(f'Val_acc: {train_history.history["val_accuracy"][runs]}\n')
        f.write(f'\nComments: {COMMENTS}\n\n')
        f.write(f'Test_loss: {loss}, Test_accuracy: {acc}\n\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('###############\n\n')
        f.write(tempfile)

    average_acc.append(acc)
    average_loss.append(loss)

print(f'Average accuracy: {np.average(average_acc)}')
print(f'Average loss: {np.average(average_loss)}')
print(average_acc)
print(average_loss)