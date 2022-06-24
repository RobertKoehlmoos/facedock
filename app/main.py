from fastapi import File, FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from .photo_processing import get_embeddings

app = FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.post("/photo")
async def photo_embeddings(file: UploadFile = File(...)):  # the = File(...) needs to be there. I can't figure out why.
    return {"result": get_embeddings(file)}
