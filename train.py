import tensorflow as tf
from tensorflow import keras


class Models:

    def __init__(self, data, epochs=5):
        (x_train, y_train), (x_test, y_test) = data
        self.x_train = x_train / 255.0
        self.x_test = x_test / 255.0
        self.y_train = y_train / 255.0
        self.y_test = y_test / 255.0

        self.epochs = epochs

        self.encoder_input = keras.Input(shape=(28, 28), name='input_image')
        x = keras.layers.Flatten()(self.encoder_input)
        self.encoder_output = keras.layers.Dense(64, activation="relu")(x)

        self.decoder_input = keras.layers.Dense(64, activation="relu")(self.encoder_output)
        x = keras.layers.Dense(784, activation="relu")(self.decoder_input)
        self.decoder_output = keras.layers.Reshape((28, 28))(x)

        self.autoencoder = keras.Model(self.encoder_input, self.decoder_output, name='autoencoder')
        self.encoder = keras.Model(self.encoder_input, self.encoder_output, name='image_encoder')
        self.decoder = keras.Model(self.decoder_input, self.decoder_output, name='image_decoder')

    def train(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
        # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.autoencoder.compile(opt, loss='mse')
        self.autoencoder.fit(
            self.x_train,
            self.x_train,
            epochs=5,
            batch_size=32,
            validation_split=0.10,
            # callbacks=[tensorboard_callback]
        )
        return self

    def save(self):
        self.autoencoder.save("models/auto_encoder.model")
        self.encoder.save("models/encoder.model")
        self.decoder.save("models/decoder.model")

    def load(self):
        self.autoencoder = keras.models.load_model("models/auto_encoder.model", compile=False)
        self.encoder = keras.models.load_model("models/encoder.model", compile=False)
        self.decoder = keras.models.load_model("models/decoder.model", compile=False)

    @staticmethod
    def load_or_train(data):
        model = Models(data)
        try:
            model.load()
            return model
        except:
            model.train()
            model.save()
            return model
