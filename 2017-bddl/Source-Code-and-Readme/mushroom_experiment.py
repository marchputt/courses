"""Final project on Big Data and Deep Learning at Xi'an Jiaotong University (2017)
Contributors: Pargorn Puttapirat (ID 3117999011), Chan Min Wai (ID 3117999104), Lee Hue (ID 3117999241)

This work is licensed under a Creative Commons Attribution 4.0 International License.
Source code is available at https://github.com/marchputt/courses/tree/master/2017-bddl
"""

# Import necessary libraries
import numpy as np          # Numpy, everyone needs Numpy!
import pandas as pd         # Panda is used to read CSV file easily.
import tensorflow as tf     # The mighty TensorFlow!
from time import time       # For the TensorBoard.

sess = tf.Session()     # Start TensorFlow session

# Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


# Hyper-parameters
batch_size = 25
epochs = 300
train_to_test_ratio = 0.8

# Load the data.
inputData = pd.read_csv('mushroom_mod.csv')
inputData = inputData[:-1]
print(inputData.shape)
inputData.head(10)   # Show first 10 lines of the data.

inputData.classes = inputData.classes.replace(to_replace=['p', 'e'], value=[0, 1])
inputData.cap_shape = inputData.cap_shape.replace(to_replace=['b', 'c', 'x', 'f', 'k', 's'], value=[0, 1, 2, 3, 4, 5])
inputData.cap_surface = inputData.cap_surface.replace(to_replace=['f', 'g', 'y', 's'], value=[0, 1, 2, 3])
inputData.cap_color = inputData.cap_color.replace(to_replace=['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
                                                    value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
inputData.bruises = inputData.bruises.replace(to_replace=['t', 'f'],value=[0, 1])
inputData.odor = inputData.odor.replace(to_replace=['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8])
inputData.gill_attachment = inputData.gill_attachment.replace(to_replace=['a', 'd', 'f', 'n'], value=[0, 1, 2, 3])
inputData.gill_spacing = inputData.gill_spacing.replace(to_replace=['c', 'w', 'd'], value=[0, 1, 2])
inputData.gill_size = inputData.gill_size.replace(to_replace=['b', 'n'], value=[0, 1])
inputData.gill_color = inputData.gill_color.replace(to_replace=['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w',
                                                                 'y'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
inputData.stalk_shape = inputData.stalk_shape.replace(to_replace=['e', 't'], value=[0, 1])
inputData.stalk_root = inputData.stalk_root.replace(to_replace=['b', 'c', 'u', 'e', 'z', 'r', '?'],
                                                    value=[0, 1, 2, 3, 4, 5, 6])
inputData.stalk_surface_above_ring = inputData.stalk_surface_above_ring.replace(to_replace=['f', 'y', 'k', 's'],
                                                                                value=[0, 1, 2, 3])
inputData.stalk_surface_below_ring = inputData.stalk_surface_below_ring.replace(to_replace=['f', 'y', 'k', 's'],
                                                                                value=[0, 1, 2, 3])
inputData.stalk_color_above_ring = inputData.stalk_color_above_ring.replace(
    to_replace=['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8])
inputData.stalk_color_below_ring = inputData.stalk_color_below_ring.replace(
    to_replace=['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8])
#inputData.veil_type = inputData.veil_type.replace(to_replace=['p','u'], value=[0,1])
inputData.veil_color = inputData.veil_color.replace(to_replace=['n', 'o', 'w', 'y'], value=[0, 1, 2, 3])
inputData.ring_number = inputData.ring_number.replace(to_replace=['n', 'o', 't'], value=[0, 1, 2])
inputData.ring_type = inputData.ring_type.replace(to_replace=['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
                                                    value=[0, 1, 2, 3, 4, 5, 6, 7])
inputData.spore_print_color = inputData.spore_print_color.replace(
    to_replace=['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8])
inputData.population = inputData.population.replace(to_replace=['a', 'c', 'n', 's', 'v', 'y'],
                                                    value=[0, 1, 2, 3, 4, 5])
inputData.habitat = inputData.habitat.replace(to_replace=['g', 'l', 'm', 'p', 'u', 'w', 'd'],
                                              value=[0, 1, 2, 3, 4, 5, 6])

X = inputData.drop(labels=['classes'], axis=1).values
y = inputData.classes.values

# set replace=False, Avoid double sampling
train_index = np.random.choice(len(X), round(len(X) * train_to_test_ratio), replace=False)

# Differentiate the training and testing data set.
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]

# Normalized processing, must be placed after the data set segmentation,
# otherwise the test set will be affected by the training set
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

# Construct the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(21,)),  # input shape required
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])

model.compile(tf.keras.optimizers.SGD(lr=0.01),
              tf.keras.losses.mean_squared_error,
              metrics=['accuracy'])

# Initialize TensorBoard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

# Train the model
model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_X, test_y),
          callbacks=[tensorboard])

# # Predict
'''Xnew = np.array([[2,1,8,0,4,3,1,0,10,0,6,1,2,7,5,3,1,4,6,5,0]])
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))'''

'''
Xnew = np.array([[2,1,8,1,4,3,1,1,10,1,6,1,2,7,5,3,1,4,6,5,1]])
ynew = model.predict(Xnew,batch_size=1)
print(ynew)
'''