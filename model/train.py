from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os

X = None
y = None

# load the dataset
def load(path = 'test-training-data.csv'):
    dataset = loadtxt(path, delimiter=',')
    # split into input (X) and output (y) variables
    global X, y
    X = dataset[:, 0:8]
    y = dataset[:, 8]

def train():
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)

    # evaluate the keras model
    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))

    save(model)

def save(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    train()
