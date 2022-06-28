from deepface import DeepFace
from retinaface import RetinaFace


# TODO: Figure out how to get the NN h5 files local so that they don't need to be downloaded from a github
def get_embeddings(photo_path: str) -> list[dict]:
    # models supported for embeddings by deepface
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
    faces = RetinaFace.extract_faces(photo_path)

    face_analysis = []
    for face in faces:
        face_analysis.append(DeepFace.analyze(face, detector_backend='skip', actions=("age", "gender", "race")))
        face_analysis[-1]['embedding'] = DeepFace.represent(face, detector_backend='skip', model_name=models[0])
    return face_analysis


if __name__ == "__main__":
    test_embeddings = get_embeddings("../test/test_people.png")
    print("There are ", len(test_embeddings), " faces in the image")
    for person in test_embeddings:
        print(f"age:{person['age']}, gender:{person['gender']}, dominant race: {person['dominant_race']}")

