from deepface import DeepFace
from retinaface import RetinaFace
from PIL import Image
import tempfile
import numpy
import zipfile
import io
import os


# TODO: Figure out how to get the various model h5 files local so that they don't need to be downloaded from a github
def analyse_photo(photo_path: str, attributes: tuple[str] = ("age", "gender", "race"),
                  embedding_requested: bool = True, model: str = "VGG-Face",
                  include_faces: bool = True) -> tuple[list[dict], io.BytesIO]:
    """
    This function takes a photo and then, based on parameters, returns some classifications and/or embedding for
    each detected face in the photo.
    :param photo_path: The location of the photo to be analysed.
    :param attributes: Classifications that will be created for each face detected
    :param embedding_requested: Determines if a facial embedding will be generated for each face
    :param model: The model used to generate the embedding for each face
    :param include_faces: Determines if the face extracted will be included with it's embedding and classifications
    :return: Returns a list where each entry includes the classifications, embedding, and face array for each face detected,
    as specified
    """
    # code for working with multiple faces taken from https://github.com/serengil/deepface/issues/321
    faces = RetinaFace.extract_faces(photo_path)

    face_analysis = []
    for face in faces:
        face_analysis.append(DeepFace.analyze(face, detector_backend='skip', actions=attributes))
        del face_analysis[-1]['region']
        if embedding_requested:
            face_analysis[-1]['embedding'] = DeepFace.represent(face, detector_backend='skip', model_name=model)
    return face_analysis, convert_photo_ndarrays_to_zip(faces if include_faces else [])


# returning multiple files taken from
# https://stackoverflow.com/questions/61163024
def convert_photo_ndarrays_to_zip(photos_arrays: list[numpy.ndarray]) -> io.BytesIO:
    """
    Takes an array of numpy arrays for pictures, such as those produced by RetinaFaces, and converts them into
    a zip folder of those arrays as photos.
    :param photos_arrays: arrays of images
    :return: An IO object containing a zip folder containing the arrays as photos
    """
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    with tempfile.TemporaryDirectory() as tmp:  # temp dir to avoid errors from answering multiple requests at once
        for i, array in enumerate(photos_arrays):
            file_location = os.path.join(tmp, f"face{i}.jpeg")
            im = Image.fromarray(array)
            im.save(file_location)

            # Add file, at correct path
            zf.write(file_location, f"face{i}.jpeg")

    # Must close zip for all contents to be written
    zf.close()
    return s


if __name__ == "__main__":
    test_embeddings = analyse_photo("../test/test_people.png")
    print("There are ", len(test_embeddings), " faces in the image")
    for person, _ in test_embeddings:
        print(f"age:{person['age']}, gender:{person['gender']}, dominant race: {person['dominant_race']}")
