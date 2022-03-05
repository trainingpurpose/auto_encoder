import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

autoencoder = keras.models.load_model("models/auto_encoder.model")
example = autoencoder.predict([x_test[0].reshape(-1, 28, 28, 1)])

plt.imshow(example[0].reshape((8, 8)), cmap="gray")

print(example[0].shape)
print(example[0])
