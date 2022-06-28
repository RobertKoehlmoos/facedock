from deepface import DeepFace
from retinaface import RetinaFace


# TODO: Figure out how to get the NN h5 files local so that they don't need to be downloaded from a github
def get_embeddings(photo_path: str, attributes: tuple[str]=("age", "gender", "race"),
                   embedding: bool = True, model: str = "VGG-Face") -> list[dict]:
    """
    This function takes a photo and configurations and then for each face it finds in the photo
    :param photo_path:
    :param attributes:
    :param embedding:
    :param model:
    :return:
    """
    faces = RetinaFace.extract_faces(photo_path)

    # todo: add a the face cutout for each face found in the photo as a file in the post reply
    face_analysis = []
    for face in faces:
        face_analysis.append(DeepFace.analyze(face, detector_backend='skip', actions=attributes))
        if embedding:
            face_analysis[-1]['embedding'] = DeepFace.represent(face, detector_backend='skip', model_name=model)
    return face_analysis


if __name__ == "__main__":
    test_embeddings = get_embeddings("../test/test_people.png")
    print("There are ", len(test_embeddings), " faces in the image")
    for person in test_embeddings:
        print(f"age:{person['age']}, gender:{person['gender']}, dominant race: {person['dominant_race']}")

