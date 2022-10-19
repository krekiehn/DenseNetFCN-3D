# init a FCN Model for training and later inference?
#from DenseNet3D import DenseNet3D_FCN
from SimpelNet import FCN_model
from data import DataGenerator, DataGenerator_performance
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import datetime

from seg_vertebra_instance_qa.__init__ import *
from seg_vertebra_instance_qa.train import *


def get_confusion_matrix(model, validation_generator):
    all_predictions = np.array([])
    all_labels = np.array([])
    for i in range(len(validation_generator)):
        x_batch, y_batch = validation_generator[i]
        predictions = model.predict(x_batch)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        labels = np.argmax(y_batch, axis = 1)
        all_labels = np.concatenate([all_labels, labels])

    return tf.math.confusion_matrix(all_predictions, all_labels)


def inference(model, val_generator, checkpoint_path=r'seg_vertebra_instance_qa\model_epoch_02_loss_0.15_acc_0.95_val_loss_0.85_val_acc_0.67.h5'):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # paths = sorted(Path(checkpoint_path).iterdir(), key=os.path.getmtime)

    model = tf.keras.models.load_model(checkpoint_path)

    #model = tf.keras.models.load_model(r'/home/sukin699/DenseNetFCN-3D/snapshots/model_epoch_86_loss_0.18_acc_0.94_val_loss_0.83_val_acc_0.68.h5')
    loss, acc = model.evaluate(val_generator, verbose=2)
    print("Restored model, val accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, val loss: {:5.2f}".format(loss))
    # except:
    #     pass

    return model


def init_model(model, checkpoint_path=r'seg_vertebra_instance_qa\model_epoch_02_loss_0.15_acc_0.95_val_loss_0.85_val_acc_0.67.h5'):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model = tf.keras.models.load_model(checkpoint_path)
    return model


if __name__ == '__main__':
    model_fcn = FCN_model(len_classes=3)
    model_fcn.summary()

    df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file), index_col='index',
                          dtype={"Ids": str, "status": int})

    batch_size = int(input('Batch size?'))*1
    # train_generator = DataGenerator(df_data, partition='train', batch_size=batch_size)
    # val_generator = DataGenerator(df_data, partition='vali', batch_size=batch_size)
    # test_generator = DataGenerator(df_data, partition='test', batch_size=batch_size)
    model_fcn = init_model(model_fcn)
    val_generator = DataGenerator_performance(df_data, partition='vali', batch_size=1)
    # model = inference(model_fcn, val_generator)

