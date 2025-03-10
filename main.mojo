from python import Python, PythonObject
from collections import Optional

def main():
    # importing works kinda weird, need to use conda instead of pip
    # and then import the module not by the name of the package but by the name of the module
    cv2 = Python.import_module("cv2")

    image = cv2.imread("test.jpg")
    print(image.shape)


    # conevert to gray to make it easier to work with
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)
    

    # conevert to gray to make it easier to work with
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)
    

    # load the haarcascade
    haarcascade_path = "haarcascade_frontalcatface.xml"
    face_classifier = cv2.CascadeClassifier(haarcascade_path)

    # detect faces
    faces_py = face_classifier.detectMultiScale(
      gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # Convert from Python list to Mojo list
    faces = List[Tuple[Int, Int, Int, Int]]()
    for f in faces_py:
        faces.append((Int(f[0]),Int(f[1]), Int(f[2]), Int(f[3])))
        
    # validate the face
    face = validate_face(faces, image)
    if face is None:
        return
    x, y, w, h = face.value()

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
def validate_face(faces: List[Tuple[Int, Int, Int, Int]], image: PythonObject, bounds: Tuple[Int, Int] = (20, 50)) -> Optional[Tuple[Int, Int, Int, Int]]:
    """
    Validate the face detected in the image. We want to make sure there is only one face
    and that the face is within the 20-50% range of the image area.
    """

    # Make sure there is only one face
    if len(faces) == 0:
        print("No faces found")
        return 
    if len(faces) > 1:
        print("Multiple faces found")
        return

    face = faces[0]
  
    # Calculate the area of the face and the image
    face_area = face[0] * face[1]
    image_area = image.shape[0] * image.shape[1]

    # Calculate the percentage of the face area relative to the image area
    face_percentage = (face_area / image_area) * 100

    # Check if the face is within the 20-50% range
    if face_percentage < bounds[0] or face_percentage > bounds[1]:
        print("Face size out of range: {face_percentage:.2f}%")
        return

    return face
