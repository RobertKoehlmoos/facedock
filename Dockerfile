FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# hard coded location for weights based on where DeepFace saves them on the docker python container
# needed for when the container is build in environments seperate from the internet
COPY ./deepface_models /root/.deepface/weights

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# below two lines taken from https://stackoverflow.com/questions/55313610
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
