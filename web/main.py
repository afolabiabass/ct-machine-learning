import json
import os
import time
import uuid
import redis
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


@app.get("/")
def index():
    return "Hello World!"


@app.post("/predict")
def predict(request: Request):
    data = {"success": False}

    if request.method == "POST":
        # Generate an ID for the classification then add the classification ID + image to the queue
        k = str(uuid.uuid4())
        # replace input with actual input from the api request
        d = {"id": k, "input": [[5, 116, 74, 0, 0, 25.6, 0.201, 30]]}
        db.rpush('query_queue', json.dumps(d))

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
        else:
            raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data
