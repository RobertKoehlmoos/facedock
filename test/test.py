from fastapi.testclient import TestClient
from app import main


client = TestClient(main.app)


def test_sanity():
    assert True


def test_get_embedding():
    with open('./test/test_people.png', 'rb') as f:
        response = client.post("/photo", files={'file': f})
    print(response.request.body[:200])
    assert len(response.text) > 0


def test_normal():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}