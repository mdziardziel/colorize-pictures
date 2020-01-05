from PIL import Image
import numpy as np


# # https://stackoverflow.com/a/47227886
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # or even "-1"

# import tensorflow as tf

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


# print("")
# print("")
# print("")
# print("")
# print("")
# print("")


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


loaded_in = load(0, 'training/grey')
loaded_out = load(0, 'training/colour')

# show_image(loaded_in)
# show_image(loaded_out, 'RGB')

normalized_in = normalize(loaded_in)
normalized_out = normalize(loaded_out)

# renormalized_in = renormalize(normalized_in)
# renormalized_out = renormalize(normalized_out)

# show_image(renormalized_in)
# show_image(renormalized_out, 'RGB')


def get_data(pics_num, path):
  data = []
  for fname in range(pics_num):
    loaded = load(fname, path)
    normalized = normalize(loaded)
    data.append(normalized)
  return np.array(data)


train_data_in = get_data(20, 'training/grey')
train_data_out = get_data(20, 'training/colour')

test_data_in = get_data(8, 'test/grey')
test_data_out = get_data(8, 'test/colour')


def build_autoencoder(in_shape, out_shape):
  # The encoder
  encoder = Sequential()
  encoder.add(InputLayer(in_shape))
  encoder.add(Flatten())
  encoder.add(Dense(np.prod(out_shape)))

  # The decoder
  decoder = Sequential()
  decoder.add(InputLayer((np.prod(out_shape),)))
  decoder.add(Dense(np.prod(out_shape))) 
  decoder.add(Reshape(out_shape))
  return encoder, decoder

INP_SHAPE = train_data_in.shape[1:] # (224, 224, 1)
OUT_SHAPE = train_data_out.shape[1:] # (224, 224, 3)

# print(INP_SHAPE)

encoder, decoder = build_autoencoder(INP_SHAPE, OUT_SHAPE)

inp = Input(INP_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(x=train_data_in, y=train_data_out, epochs=2, validation_data=[test_data_in, test_data_out])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()