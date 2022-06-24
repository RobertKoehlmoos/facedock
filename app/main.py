from fastapi import File, FastAPI, UploadFile
from photo_processing import get_embeddings

app = FastAPI()


@app.get("/")
def read_root():
    return {"msg": "Hello World"}


@app.post("/photo")
async def photo_embeddings(file: UploadFile = File(...)):  # the = File(...) needs to be there. I can't figure out why.
    return {"result": get_embeddings(file)}
