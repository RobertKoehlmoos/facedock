from fastapi import File, FastAPI, UploadFile, Form, HTTPException, Response
from fastapi.responses import RedirectResponse
from .photo_processing import analyse_photo
import tempfile
import aiofiles
import os
import json


app = FastAPI(
    title="Facedock"
)


@app.get("/")
async def read_root():
    return RedirectResponse(url="/docs")


@app.post("/photo", summary="Submit a photo and select attributes to be extracted from models, optionally returning cut out faces.")
async def photo_embeddings(photo: UploadFile = File(...),
                           attributes: list[str] = Form(["age", "gender", "race", "embedding"]),  # mutable default
                           model: str = Form("VGG-Face"), return_faces: bool = Form(True)):
    """
    Takes a single photo and provides back zip of all the faces found in the photo, along with a selection of
    classifications and/or an embedding for each face. The zip will contain the faces labelled face{i}.jpeg, where i
    corresponds to their face's index within the results list. The classifications and embedding are found in the
    headers under "results" as a json string.
    :param photo: The image containing the faces.
    :param attributes: A list of attributes that models can extract from each face, e.g. age, gender
    :param model: The name of a model to use to generate the embedding
    :param return_faces: If the faces should be returned in a zip. Otherwise the zip file will be empty.
    """
    valid_deepface_models = ("VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace")
    if model not in valid_deepface_models:
        raise HTTPException(status_code=422, detail=f"Invalid model name '{model}'. Valid model are {valid_deepface_models}.")

    valid_deepface_attributes = ('age', 'gender', 'race', 'emotion', 'embedding')
    if invalid_attributes := tuple(attribute for attribute in attributes if attribute not in valid_deepface_attributes):
        raise HTTPException(status_code=422,
                            detail=f"Invalid attribute(s) {invalid_attributes}. Valid attributes are {valid_deepface_attributes}.")

    # checking if the user requested an embedding, and then separating it from the attributes for the analysis
    if get_embedding := "embedding" in attributes:
        attributes = tuple(attribute for attribute in attributes if attribute != "embedding")

    # temp directory to avoid error when receiving photos with the same name
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, photo.filename)
        async with aiofiles.open(path, "wb+") as temp_photo:
            await temp_photo.write(photo.file.read())
        classifications, faces_zip = analyse_photo(path, attributes=tuple(attributes),
                                                   embedding_requested=get_embedding, model=model,
                                                   include_faces=return_faces)
        resp = Response(faces_zip.getvalue(), media_type="application/x-zip-compressed",
                        headers={'Content-Disposition': 'attachment;filename=faces.zip',
                                 "results": json.dumps(classifications)})

    return resp
