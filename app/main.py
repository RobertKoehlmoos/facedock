from fastapi import File, FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse
from .photo_processing import get_embeddings
import tempfile
import aiofiles
import os

app = FastAPI()


@app.get("/")
async def read_root():
    return RedirectResponse(url="/docs")


@app.post("/photo")
async def photo_embeddings(photo: UploadFile = File(...),
                           attributes: list[str] = Form(["age", "gender", "race"]),
                           embedding: bool = Form(True),
                           model: str = Form("VGG-Face")):
    valid_deepface_models = ("VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace")
    if model not in valid_deepface_models:
        raise HTTPException(status_code=422, detail=f"Invalid model name. Valid model are {valid_deepface_models}.")

    valid_deepface_attributes = ('age', 'gender', 'race', 'emotion')
    if any(attribute not in valid_deepface_attributes for attribute in attributes):
        raise HTTPException(status_code=422,
                            detail=f"Invalid attribute(s). Valid attributes are {valid_deepface_attributes}.")

    # temp directory to avoid error when receiving photos with the same name
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, photo.filename)
        async with aiofiles.open(path, "wb+") as temp_photo:
            await temp_photo.write(photo.file.read())
        result = get_embeddings(path, attributes=tuple(attributes), embedding=embedding, model=model)
    return {"result": result}
