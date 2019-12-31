from keras.models import model_from_json
import json
import os
import sys
import time
import numpy as np
import redis

# Connect to Redis server
db = redis.StrictRedis(host='localhost')

model = None

def load():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

def process():
    # Continually poll for new input to predict
    load()
    while True:
        # Pop off multiple images from Redis queue atomically
        with db.pipeline() as pipe:
            # queue = []
            pipe.lrange('query_queue', 0, int(32) - 1)
            pipe.ltrim('query_queue', int(32), -1)
            queue, _ = pipe.execute()

        for q in queue:
            # Deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            input = np.array(q['input'])
            key = q['id']

            result = model.predict(input)

            print("Printing result")

            db.set(key, json.dumps(result.tolist()))

        # Sleep for a small amount
        time.sleep(float(0.25))


if __name__ == "__main__":
    process()
