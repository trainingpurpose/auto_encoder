from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense

model = Sequential()
model.add(Conv2D(128, (4, 4), input_shape=(6, 7, 1)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))


