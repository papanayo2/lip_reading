from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt
from process_data import get_features

num_classes = 100

def normalize_data(matrix):
    """
        Normalizes the data so it is between 0-1.
    """
    new = np.zeros((len(matrix), len(matrix[0])), dtype=np.float32)
    maximum = matrix[:].max()
    minimum = matrix[:].min()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i,j] = (matrix[i,j] - minimum)/(maximum - minimum)
            
    return new

def make_labels(training_set):
    """
        Creates one hot labels for data.
    """
    
    one_hot = np.zeros((len(training_set), num_classes), dtype=np.uint8)
    
    j = 0
    for i in range(len(training_set)):
        if j == 100:
            j = 0
            
        temp = np.zeros((num_classes), dtype=np.uint8)
        temp[j] = 1
        j += 1
        one_hot[i] = temp

    return one_hot

np.set_printoptions(threshold=np.nan)

train_x = np.load('training2.npy')
train_x = normalize_data(train_x)

train_y = make_labels(train_x)


test_x = np.load('testing2.npy')
test_x = normalize_data(test_x)
test_y = make_labels(test_x)
 
epochs = 100
batch_size = 100
#------------------------------------------------------------------------
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(10,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

h1 = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
score = model.evaluate(test_x, test_y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])





file_name = '' #enter test video here to make prediction on
data = process_data.get_features(file_name)
data = normalize_data(data[0])
print(data.shape)
res = model.predict(data.reshape(1,10))
print(res)