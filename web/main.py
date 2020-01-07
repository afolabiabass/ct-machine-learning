import json
import os
import time
import uuid
import redis
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


@app.get("/")
def index():
    return "Hello World!"


@app.get("/predict")
def predict(request: Request, count: int = 0, more: int = 0, keyword: int = 0, rate: int = 0):
    data = {"success": False}

    if count == 0 and more == 0 and keyword == 0 and rate == 0:
        return data

    # Generate an ID for the classification then add the classification ID + request
    k = str(uuid.uuid4())
    # replace input with actual input from the api request
    d = {"id": k, "input": [[count, more, keyword, rate]]}
    db.rpush(os.environ.get("QUERY_QUEUE"), json.dumps(d))

    # Keep looping for CLIENT_MAX_TRIES times
    num_tries = 0
    while num_tries < CLIENT_MAX_TRIES:
        num_tries += 1

        # Attempt to grab the output predictions
        output = db.get(k)

        # Check to see if our model has classified the input image
        if output is not None:
            # Add the output predictions to our data dictionary so we can return it to the client
            output = output.decode("utf-8")
            data["predictions"] = json.loads(output)

            # Delete the result from the database and break from the polling loop
            db.delete(k)
            break

        # Sleep for a small amount to give the model a chance to classify the input image
        time.sleep(float(os.environ.get('CLIENT_SLEEP')))

        # Indicate that the request was a success
        data["success"] = True


    # Return the data dictionary as a JSON response
    return data
