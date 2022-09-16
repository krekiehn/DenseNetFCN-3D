# init a FCN Model for training and later inference?
#from DenseNet3D import DenseNet3D_FCN
from SimpelNet import FCN_model
from data import DataGenerator
import tensorflow as tf
import pandas as pd
import os
import datetime

from seg_vertebra_instance_qa.__init__ import *


def train(model, train_generator, val_generator, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path,
                              'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_loss_{val_loss:.2f}_val_acc_{val_accuracy:.2f}.h5')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    # try:
    #latest = tf.train.latest_checkpoint(checkpoint_path)
    #print(latest)
    checkpoint_path_2 = '../snapshots'
    # model.load_weights(checkpoint_path)

    # search_dir = "/mydir/"
    # os.chdir(checkpoint_path)
    print(os.listdir())
    print(checkpoint_path,os.path.isdir(checkpoint_path))
    print(checkpoint_path_2, os.path.isdir(checkpoint_path_2))
    files = filter(os.path.isfile, os.listdir(checkpoint_path_2))
    files = [os.path.join(checkpoint_path_2, f) for f in files]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    print(os.path.join(checkpoint_path_2, files[-1]))
    print(os.path.isfile(os.path.join(checkpoint_path_2, files[-1])))

    model = tf.keras.models.load_model(os.path.join(checkpoint_path_2, files[-1]))

    #model = tf.keras.models.load_model(r'/home/sukin699/DenseNetFCN-3D/snapshots/model_epoch_86_loss_0.18_acc_0.94_val_loss_0.83_val_acc_0.68.h5')
    loss, acc = model.evaluate(val_generator, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    # except:
    #     pass

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  callbacks=[callback_checkpoint, callback_tensorboard, callback_early_stopping],
                                  validation_data=val_generator,
                                  validation_steps=len(val_generator))

    model_dir = r'./saved_model'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f'model_{datetime.datetime.now().strftime("%Y%m%d")}'))

    return history


if __name__ == '__main__':
    # model_fcn = DenseNet3D_FCN((None, None, None, 1), nb_dense_block=5, growth_rate=16,
    #                                nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid')
    # model_fcn.summary()

    # multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model_fcn = FCN_model(len_classes=3)
    model_fcn.summary()

    df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file), index_col='index',
                          dtype={"Ids": str, "status": int})

    batch_size = int(input('Batch size?'))*3
    train_generator = DataGenerator(df_data, partition='train', batch_size=batch_size)
    val_generator = DataGenerator(df_data, partition='vali', batch_size=batch_size)
    test_generator = DataGenerator(df_data, partition='test', batch_size=batch_size)

    h = train(model_fcn, train_generator, val_generator, epochs=1000)

