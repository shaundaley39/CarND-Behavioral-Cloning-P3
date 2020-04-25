import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

rundir='merged'
batch_size=32


samples=[]
with open(rundir+'/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
      samples.append([line[0].split('/')[-1], line[3], True])
      samples.append([line[0].split('/')[-1], line[3], False])
shuffle(samples)
print("Number of samples: "+str(len(samples))+", to be used in an 80/20 training/validation split")
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = rundir + '/IMG/' + batch_sample[0]
                image = cv2.imread(name)
                if(batch_sample[2]):
                    images.append(np.fliplr(image))
                    angles.append(-1.0*float(batch_sample[1]))
                else:
                    images.append(image)
                    angles.append(float(batch_sample[1]))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


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
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save('model.h5')
