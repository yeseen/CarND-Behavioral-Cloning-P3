import csv
import cv2
import numpy as np


images =[]
angles =[]
with open('dataX/driving_log.csv') as csvfile:
	for line in csv.reader(csvfile):
		s_path = line[0]
		fname = s_path.split('\\')[-1]
		c_path = 'dataX/IMG/' + fname
		print(c_path)
		image = cv2.imread(c_path)
		try :
			image_flip = np.fliplr(image)
			#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			images.append(image)
			angles.append(float(line[3]))
			#image = cv2.flip(image,1)
			images.append(image_flip)
			angles.append( - float(line[3]))
		except :
		 	pass


X_train = np.array(images)
y_train = np.array(angles)

print(X_train.shape)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Conv2D
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))


model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid'))

model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid'))

model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid'))

model.add(Conv2D(64, 3, 3, border_mode='valid'))

model.add(Conv2D(64, 3, 3, border_mode='valid'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('modelX1.h5')
exit()

