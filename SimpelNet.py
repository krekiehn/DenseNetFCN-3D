import tensorflow as tf


def FCN_model(len_classes=5, dropout_rate=0.2, shape=(None, None, None, 1)):
    input = tf.keras.layers.Input(shape=shape)

    x0, x1 = tf.split(input, num_or_size_splits=2, axis=-1)
    x1 = tf.keras.layers.GaussianNoise(stddev=1.0)(x1)
    x = tf.keras.layers.Concatenate()([x0, x1])

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=9, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling3D()(x)

    x = tf.keras.layers.Conv3D(filters=16*8, kernel_size=5, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling3D()(x)

    x = tf.keras.layers.Conv3D(filters=16*8*2, kernel_size=3, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling3D()(x)

    x = tf.keras.layers.Conv3D(filters=16*8*4, kernel_size=3, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling3D()(x)

    # x = tf.keras.layers.Conv3D(filters=16*8*8, kernel_size=1, strides=1)(x)
    # #x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Uncomment the below line if you're using dense layers

    #x = tf.keras.layers.GlobalMaxPooling3D()(x)  # GlobalAverage oder GlobalMax?
    x = tf.keras.layers.GlobalAveragePooling3D()(x)  # GlobalAverage oder GlobalMax?
    # x = tf.keras.layers.GlobalMaxPooling3D()(x)

    # Fully connected layer 1
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=64)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 1
    # x = tf.keras.layers.Conv3D(filters=64, kernel_size=1, strides=1)(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 2
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=len_classes)(x)
    # predictions = tf.keras.layers.Activation('softmax')(x)

    # Fully connected layer 2
    # x = tf.keras.layers.Conv3D(filters=len_classes, kernel_size=1, strides=1)(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    predictions = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)

    print(model.summary(line_length=150))
    print(f'Total number of layers: {len(model.layers)}')

    return model


if __name__ == "__main__":
    model = FCN_model(len_classes=5, dropout_rate=0.2, shape=(None,None,None,2))