import tensorflow as tf
from tensorflow import keras
from datetime import datetime

epochs = 5
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

encoder_input = keras.Input(shape=(28, 28, 1), name='input_image')
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation="relu")(x)

# encoder = keras.Model(encoder_input, encoder_output, name='image_encoder')
decoder_input = keras.layers.Dense(64, activation="relu")(encoder_output)

x = keras.layers.Dense(784, activation="relu")(decoder_input)
decoder_output = keras.layers.Reshape((28, 28, 1))(x)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()

autoencoder.compile(opt, loss='mse')

autoencoder.fit(
        x_train,
        x_train,
        epochs=5,
        batch_size=32,
        validation_split=0.10,
        callbacks=[tensorboard_callback]
    )

autoencoder.save("models/auto_encoder.model")
