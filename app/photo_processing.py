from deepface import DeepFace
from retinaface import RetinaFace
from PIL import Image
import tempfile
import numpy
import zipfile
import io
import os


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
        face_analysis.append(DeepFace.analyze(face, detector_backend='skip', actions=attributes, prog_bar=False))
        del face_analysis[-1]['region']
        if embedding_requested:
            face_analysis[-1]['embedding'] = DeepFace.represent(face, detector_backend='skip', model_name=model)
    if not faces:
        # if no faces are detected assume the whole image is a face, this is a known issue when using pre-cut out faces from different models
        face_analysis.append(DeepFace.analyze(photo_path, detector_backend='skip', actions=attributes, prog_bar=False,
                                              enforce_detection=False))
        # these three lines are repeated from above, should turn them into a function if the repeated section gets any longer
        del face_analysis[-1]['region']
        if embedding_requested:
            face_analysis[-1]['embedding'] = DeepFace.represent(photo_path, detector_backend='skip', model_name=model)
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
    print("There are ", len(test_embeddings[0]), " faces in the image")
    for person in test_embeddings[0]:
        print(f"age:{person['age']}, gender:{person['gender']}, dominant race: {person['dominant_race']}")
