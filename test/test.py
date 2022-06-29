from fastapi.testclient import TestClient
from app import main


client = TestClient(main.app)


def test_sanity():
    assert True


def test_get_embedding():
    with open('./test/test_people.png', 'rb') as f:
        response = client.post("/photo", files={'photo': f})
    assert len(response.content) > 0
    assert "results" in response.headers


def test_invalid_model_name():
    with open('./test/test_people.png', 'rb') as f:
        response = client.post("/photo", files={'photo': f}, data={'model': 'invalid name'})
    print(response.text)
    print(response.status_code)
    assert response.status_code == 422


def test_normal():
    response = client.get("/")
    print(response.text)
    assert response.status_code == 200


if __name__ == "__main__":
    test_get_embedding()
