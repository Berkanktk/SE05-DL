import tensorflow as tf


# Definer function til at bygge en simpel Keras (funktionel) CNN model
def cnn_model(input_shape=None, n_layers=5, n_units=[32, 64, 128, 256, 512], kernel_size=(3, 3), kernel_activation='relu',
              output_actvation='softmax', batchnorm=False, droprate=0.25, num_classes=7):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(input)

    for i in range(n_layers):
        x = tf.keras.layers.Conv2D(filters=n_units[i], kernel_size=kernel_size, activation=kernel_activation, padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        if batchnorm == True:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(rate=droprate)(x)  # randomly dropping out a certain proportion of the nodes

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation=output_actvation)(x)  # Final layer

    return tf.keras.Model(inputs=input, outputs=output)
