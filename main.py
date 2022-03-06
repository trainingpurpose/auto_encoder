import tensorflow as tf
from tensorflow import keras
from train import Models

data = tf.keras.datasets.mnist.load_data()
a = Models(data=data)

# a.load()
a.train()
# a.save()

autoencoder_prediction = a.encoder.predict([a.x_test[0].reshape(-1, 28, 28, 1)])

print(autoencoder_prediction)
