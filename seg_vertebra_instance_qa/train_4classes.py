# init a FCN Model for training and later inference?
#from DenseNet3D import DenseNet3D_FCN
from SimpelNet import FCN_model
from data import DataGenerator_4_classes
import tensorflow as tf
import pandas as pd
import os
from pathlib import Path
import datetime

from seg_vertebra_instance_qa.__init__ import *

parameters = {
    'n_channels': 2, 
    'n_classes': 4
}

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

    paths = sorted(Path(checkpoint_path).iterdir(), key=os.path.getmtime)

    if len(paths) > 0:
        model = tf.keras.models.load_model(paths[-1])
    
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
    if multi_gpu_flag:
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model_fcn = FCN_model(len_classes=3)
        #model_fcn.summary()
    else:
        # non mutli GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        model_fcn = FCN_model(len_classes=parameters['n_classes'], dropout_rate=0.2, shape=(None, None, None, parameters['n_channels']))

    df_data = pd.read_csv(os.path.join(dataframe_file_path, dataframe_file_4classes), index_col='index')

    batch_size = int(input('Batch size?'))*4
    train_generator = DataGenerator_4_classes(df_data, partition='train', batch_size=batch_size, n_channels=2, n_classes=4)
    val_generator = DataGenerator_4_classes(df_data, partition='vali', batch_size=batch_size, n_channels=2, n_classes=4)
    test_generator = DataGenerator_4_classes(df_data, partition='test', batch_size=batch_size, n_channels=2, n_classes=4)

    #h = train(model_fcn, train_generator, val_generator, epochs=1)

