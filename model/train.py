from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
import csv
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path

# env_path = Path('..') / '.env'
# load_dotenv(dotenv_path=env_path)
load_dotenv()

X = None
y = None


# load the dataset
def loadFromFile(path='test-training-data-2.csv'):
    dataset = loadtxt(path, delimiter=',')
    # split into input (X) and output (y) variables
    global X, y
    X = dataset[:, 0:4]
    y = dataset[:, 4]


def loadFromDb(path='test-training-data-2.csv'):
    try:
        mydb = mysql.connector.connect(
            host=os.environ.get('DB_HOST'),
            port=os.environ.get('DB_PORT'),
            user=os.environ.get('DB_USERNAME'),
            passwd=os.environ.get('DB_PASSWORD'),
            database=os.environ.get('DB_DATABASE')
        )

        mycursor = mydb.cursor()
        mycursor.execute(os.environ.get('TRAINING_DATA_SQL_QUERY'))
        data = mycursor.fetchall()

        with open(path, "w", newline='') as csv_file:
            c = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Iterate over `data` and write to the csv file
            for row in data:
                c.writerow(row)

        dataset = loadtxt(path, delimiter=',')
        # split into input (X) and output (y) variables
        global X, y
        X = dataset[:, 0:4]
        y = dataset[:, 4]
    except:
        print('Could not load data from database')


def train():
    try:
        # define the keras model
        loadFromDb()
        model = Sequential()
        model.add(Dense(12, input_dim=4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X, y, epochs=150, batch_size=10, steps_per_epoch=None)

        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy * 100))

        save(model)
    except:
        print('Error occurred while trying to train data')


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
