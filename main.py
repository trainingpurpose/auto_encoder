import tensorflow as tf
from matplotlib import pyplot as plt
from train import Models

data = tf.keras.datasets.mnist.load_data()
a = Models.load_or_train(data=data)

autoencoder_prediction = a.encoder.predict([a.x_test[0].reshape(-1, 28, 28, 1)])

ae_out = a.autoencoder.predict([a.x_test[1].reshape(-1, 28, 28, 1)])
e_out = a.encoder.predict([a.x_test[1].reshape(-1, 28, 28, 1)])
d_out = a.decoder.predict([e_out])

plt.ioff()
fig = plt.figure(figsize=(10, 7))

rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(a.x_test[1], cmap="gray")
plt.axis("off")
plt.title("Original")

fig.add_subplot(rows, columns, 2)
plt.imshow(e_out[0].reshape(8, 8), cmap="gray")
plt.axis("off")
plt.title("Encoded")

fig.add_subplot(rows, columns, 3)
plt.imshow(d_out[0], cmap="gray")
plt.axis("off")
plt.title("Decoded")

fig.add_subplot(rows, columns, 4)
plt.imshow(ae_out[0], cmap="gray")
plt.axis("off")
plt.title("AutoEncoded")
