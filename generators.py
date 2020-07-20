import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


rundir='merged'
batch_size=16

samples=[]
with open(rundir+'/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
      for flip in [True, False]:
          samples.append([line[0].split('/')[-1], float(line[3]), flip])
          samples.append([line[1].split('/')[-1], float(line[3])+0.2, flip])
          samples.append([line[2].split('/')[-1], float(line[3])-0.2, flip])
shuffle(samples)
# print("Number of samples: "+str(len(samples))+", to be used in an 80/20 training/validation split")
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
                    angles.append(-1.0*batch_sample[1])
                else:
                    images.append(image)
                    angles.append(batch_sample[1])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
