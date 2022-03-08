import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = data
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = y_train / 255.0
y_test = y_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=20
)

# model.save("models/mnist.model")

# mnist_model = keras.models.load_model("models/mnist.model", compile=False)

sample = x_test[0]
reshaped_sample = x_test[0].reshape(-1, 28, 28)

autoencoder_prediction = model.predict(reshaped_sample)

print(y_test)
