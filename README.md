# facedock - Facial Recognition as a Service! [![Validation Tests](https://github.com/RobertKoehlmoos/facedock/actions/workflows/actions.yml/badge.svg)](https://github.com/RobertKoehlmoos/facedock/actions/workflows/actions.yml)  
facedock is a web service that enables users to send in photos and receive back
classifications and embeddings for the faces in those photos.
## Installation
facedock is intended to be run as a docker container.  
You must first install docker on your computer, which can be found at
https://docs.docker.com/get-docker/  
Then follow the below instructions:
1. Open a command prompt in the root folder of this directory and run
the command `docker build -t facedock .`
This will create a docker image of facedock on your local computer. You
can confirm this by running `docker images` which will include `facedock` if the
build was successful.  NOTE: building the image can take a while due
   installing many large computer vision libraries. On a computer with 8Gb of RAM
   and 60Mb/s download speed, building took 444 seconds, or about 7 minutes.
2. This image can then be run using the command `docker run -p 80:80 facedock`.
Make sure your local routing policy and firewalls allows incoming connection. A
   port other than 80 can be specified by replacing the 80 following the colon in
   the provided command.
3. Once running, you should see output similar to below in your console.
```console
\Users\adora\facedock>docker run -p 80:80 facedock
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```
facedock can be run locally without docker using the python installation native
to your machine. To do this, follow the below instructions.
1. Install python with a verison of at least 3.10 on your local computer. 
   This can be installed from https://www.python.org/downloads/
2. Run the command `pip install --no-cache-dir --upgrade -r /code/requirements.txt`
Similar to before, this can take some time to install.
3. Open a console at the root directory of facedock.
4. Run the command `python -m uvicorn app.main:app --host 0.0.0.0 --port 80`
5. Confirm it is running by opening a browser and navigating to `localhost`.
This should redirect you to a documentation page.
Console output should look similar to below.
```console
\Users\adora\facedock>python -m uvicorn app.main:app --host 0.0.0.0 --port 80
INFO:     Started server process [18648]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
INFO:     127.0.0.1:60974 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     127.0.0.1:60974 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:60974 - "GET /openapi.json HTTP/1.1" 200 OK
```

## How to Use
To use facedock, send an HTTP POST request to the machine and port where facedock
is hosted with the url `/photo`.  
facedock expects an image to be sent with the post request.  
facedock returns a zip folder containing the images of each face extracted from
the photo, named `face{i}.jpeg`, where {i} is replaced with a number.
In the headers of the response, there is a `results` tag. This tag will contain
a json encoded string. Decoding this string will return a list of maps containing
various attributes and/or encodings for each face. The index of each entry in
the list corresponds the image in labelled with the same number.  
By default, each map contains the age, gender, breakdown of associations with
races, the dominant race, and an embedding of that face generated using the 
model VGG-Face.  

Optional parameters include:  
attributes - A list containing attributes to be generated for each face. Valid
attributes are 'age', 'gender', 'race', 'emotion', and 'embedding'.  
model - The model to generate the embedding. Valid models are 
"VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", and "ArcFace".  
return_faces: If the extracted faces should be returned in a zip. 
Otherwise the zip file will be empty.
## Usage Examples
Download the faces zip and save the classifications to a text file.
```python
import requests
with open('people.png', 'rb') as photo:
    response = requests.post('http://facedock.com/photo', files={'photo': photo})
with open("faces.zip", "wb") as faces:
    faces.write(response.content)
with open("results.txt", "w") as results:
    results.write(response.headers['results'])
```
Request only the ages and genders of the people in the photo and save them
locally.
```python
import requests
with open('people.png', 'rb') as photo:
    response = requests.post('http://facedock.com/photo', files={'photo': photo},
                             data={'attributes': ['age', 'gender'], 
                                   'return_faces': False})
with open("results.txt", "w") as results:
    results.write(response.headers['results'])
```
Request only embeddings using the ArcFace model and convert them to a list of
python dictionaries.
```python
import requests
import json
with open('people.png', 'rb') as photo:
    response = requests.post('http://facedock.com/photo', files={'photo': photo},
                             data={'attributes': ['embedding'], 
                                   'model': 'ArcFace', 'return_faces': False})
emebeddings = json.loads(response.header['results'])
```