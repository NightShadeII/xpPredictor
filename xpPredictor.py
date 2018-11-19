from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)

# Load the dataset
dataset = numpy.loadtxt("experience.csv", delimiter=",")
numpy.random.shuffle(dataset)

# Set to first and second column
X = dataset[: ,0]
experienceValues = dataset[: ,1]

# Fit all the X values between 0 and 1
X = numpy.array(X)
X = X/23

# Fit Y into buckets [0 to 100,000), [0 to 200,000),
# [900,000 to 1,000,000]

Y = []
default = [0] * 10
for item in experienceValues:
    bucket = list(default)
    bucket[ int(min(item // 100000, 9)) ] = 1
    Y.append(bucket)
Y = numpy.array(Y)

# Construct and compile the model

model = Sequential()
model.add(Dense(64, activation = 'relu', input_dim = 1))
model.add(Dense(32, activation = 'relu', input_dim = 1))
model.add(Dense(units = 10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Fit the model and run predictions on the dataset

model.fit(X, Y, epochs=200, batch_size=10, verbose = 2)

predictions = model.predict(X)

print([numpy.round(y) for y in predictions])

# input so I can see the output on console window

input()

