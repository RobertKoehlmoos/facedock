from fastapi import File, FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from .photo_processing import get_embeddings
import aiofiles
import os

app = FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.post("/photo")
async def photo_embeddings(photo: UploadFile = File(...)):  # the = File(...) needs to be there. I can't figure out why.
    # todo: choose locations that won't cause problems when multiple people make the same api call at once
    temp_photo_location = f"./{photo.filename}"
    async with aiofiles.open(temp_photo_location, "wb+") as temp_photo:
        temp_photo.write(photo.file.read())
    result = get_embeddings(temp_photo_location)
    os.remove(temp_photo_location)
    return {"result": result}
