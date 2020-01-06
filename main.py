from PIL import Image
import numpy as np

from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

import matplotlib.pyplot as plt

TRAINING_PICTURES = 7727
TEST_PICTURES = 3271
EXTENSION = "jpg"

# based on https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

def load(fname, path) :
  img = Image.open(path + "/" + str(fname) + "." + EXTENSION)
  return np.array(img)

def normalize(data):
  return (data / 255) - 0.5

def renormalize(data):
  return (data + 0.5) * 255

def show_image(data, mode='L'):
  int_converted = np.clip(data.astype('uint8'), 0, 255)
  img = Image.fromarray(int_converted, mode)
  img.show()
  return

def get_data(pics_num, path):
  data = []
  for fname in range(pics_num):
    loaded = load(fname, path)
    normalized = normalize(loaded)
    data.append(normalized)
  return np.array(data)


train_data_in = get_data(7000, 'training/grey')
train_data_out = get_data(7000, 'training/colour')

test_data_in = get_data(3000, 'test/grey')
test_data_out = get_data(3000, 'test/colour')

def build_autoencoder(in_shape, out_shape):
  decoder = Sequential()
  decoder.add(InputLayer(in_shape))
  decoder.add(Flatten())
  decoder.add(Dense(np.prod(out_shape))) 
  decoder.add(Reshape(out_shape))
  return decoder

INP_SHAPE = train_data_in.shape[1:] # (224, 224, 1)
OUT_SHAPE = train_data_out.shape[1:] # (224, 224, 3)

decoder = build_autoencoder(INP_SHAPE, OUT_SHAPE)

inp = Input(INP_SHAPE)
reconstruction = decoder(inp)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(x=train_data_in, y=train_data_out, epochs=10, validation_data=[test_data_in, test_data_out])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()