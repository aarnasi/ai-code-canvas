import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define a simple neural network model with one dense layer
# The model has a single neuron (unit) and takes a single input value
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model using Stochastic Gradient Descent (SGD) optimizer
# and Mean Squared Error (MSE) as the loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define training data (xs as iclnput features, ys as corresponding outputs)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the model for 500 epochs to learn the pattern in the data
model.fit(xs, ys, epochs=500)

# Make a prediction using the trained model
# Predicts the output for an input of 10.0
print(model.predict(np.array([10.0])))
