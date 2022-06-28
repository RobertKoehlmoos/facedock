from fastapi import File, FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from .photo_processing import get_embeddings
import tempfile
import aiofiles
import os
temp = tempfile.TemporaryFile()
print(temp)
print(temp.name)
app = FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.post("/photo")
async def photo_embeddings(photo: UploadFile = File(...)):  # the = File(...) needs to be there. I can't figure out why.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, photo.filename)
        async with aiofiles.open(path, "wb+") as temp_photo:
            await temp_photo.write(photo.file.read())
        result = get_embeddings(path)
    return {"result": result}
