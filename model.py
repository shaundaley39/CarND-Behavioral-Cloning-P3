from generators import *
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# MODEL
model = Sequential()
model.add(Cropping2D(cropping=((65,30),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(300)) # 150
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

def reset_weights(model):
  session = K.get_session()
  for layer in model.layers: 
    if hasattr(layer, 'kernel_initializer'):
      layer.kernel.initializer.run(session=session)


batch_size=256
epochs=10
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit_generator(train_generator,
                              steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                              validation_data=validation_generator,
                              validation_steps=np.ceil(len(validation_samples)/batch_size),
                              epochs=epochs,
                              verbose=1,
                              callbacks=[checkpoint])
print("best loss: "+str(history.history['loss'][-1]))
