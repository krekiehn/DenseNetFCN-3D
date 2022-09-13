# init a FCN Model for training and later inference?
#from DenseNet3D import DenseNet3D_FCN
from SimpelNet import FCN_model
from data import DataGenerator
import tensorflow as tf
import pandas as pd
import os

from seg_vertebra_instance_qa.__init__ import *


def train(model, train_generator, val_generator, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path,
                              'model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{acc:.2f}_val_loss_{val_loss:.2f}_val_acc_{val_acc:.2f}.h5')

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                                                save_best_only=True, verbose=1)],
                                  validation_data=val_generator,
                                  validation_steps=len(val_generator))

    return history


if __name__ == '__main__':
    # model_fcn = DenseNet3D_FCN((None, None, None, 1), nb_dense_block=5, growth_rate=16,
    #                                nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid')
    # model_fcn.summary()
    model_fcn = FCN_model(len_classes=3)
    model_fcn.summary()

    df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file), index_col='index',
                          dtype={"Ids": str, "status": int})

    batch_size = int(input('Batch size?'))*3
    train_generator = DataGenerator(df_data, partition='train', batch_size=batch_size)
    val_generator = DataGenerator(df_data, partition='vali', batch_size=batch_size)
    test_generator = DataGenerator(df_data, partition='test', batch_size=batch_size)

    train(model_fcn, train_generator, val_generator, epochs=1)

