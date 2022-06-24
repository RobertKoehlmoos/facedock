import requests, pytest


def test_get_embedding():
    with open('test_people.png', 'rb') as f:
        response = requests.post('http://localhost/photo', files={'file': f})
    print(response.text)
    print(response.request.body[:200])


def test_normal():
    response = requests.get("http://localhost")
    print(response.text)

test_get_embedding()