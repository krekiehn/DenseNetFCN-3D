import tensorflow as tf


def FCN_model(len_classes=5, dropout_rate=0.2, shape=(None, None, None, 1)):
    input = tf.keras.layers.Input(shape=shape)

    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same')(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling3D()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.MaxPooling3D()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.MaxPooling3D()(x)

    # x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=1)(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.MaxPooling3D()(x)

    # x = tf.keras.layers.Conv3D(filters=512, kernel_size=3, strides=1)(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Uncomment the below line if you're using dense layers
    # x = tf.keras.layers.GlobalMaxPooling3D()(x)

    # Fully connected layer 1
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(units=64)(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 1
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 2
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(units=len_classes)(x)
    # predictions = tf.keras.layers.Activation('softmax')(x)

    # Fully connected layer 2
    x = tf.keras.layers.Conv3D(filters=len_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling3D()(x)
    predictions = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)

    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model


if __name__ == "__main__":
    model = FCN_model(len_classes=5, dropout_rate=0.2)