import cv2
import numpy as np
from pathlib import Path

# Make path system-independent
fd_model_path = Path(__file__).parent.parent / "datasources" / "face_detection_model" / "face_detection_yunet_2023mar.onnx"
fd_model_threshold = 0.5

fr_model_path = Path(__file__).parent.parent / "datasources" / "face_recognition_model" / "face_recognition_sface_2021dec.onnx"
fr_model_threshold = 0.5

known_faces_dict = {}

def visualize(input, faces, thickness=2):
    # faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]
    # 0-1: x, y of bbox top left corner
    # 2-3: width, height of bbox
    # 4-5: x, y of right eye (blue point in the example image)
    # 6-7: x, y of left eye (red point in the example image)
    # 8-9: x, y of nose tip (green point in the example image)
    # 10-11: x, y of right corner of mouth (pink point in the example image)
    # 12-13: x, y of left corner of mouth (yellow point in the example image)
    # 14: face score
    if faces is not None:
        for idx, face in enumerate(faces):
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
    
def visualize_recognized_faces(input, faces, recognized_faces, fps):
    # 14: face score
    if faces is not None:
        for idx, face in enumerate(faces):
            coords = face[:-1].astype(np.int32)
            if recognized_faces[idx] is not None:
                cv2.putText(input, recognized_faces[idx].upper(), (coords[0], coords[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_features(img, face, face_recognizer = None):
    if face_recognizer is None:
        face_recognizer = cv2.FaceRecognizerSF.create(
            model=str(fr_model_path),
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
    
    #  align crop face
    aligned_face = face_recognizer.alignCrop(img, face)
    
    # get features from face recognizer
    features = face_recognizer.feature(aligned_face)
    return features

def recognize_faces(img, faces):
    # initialize face recognizer
    face_recognizer = cv2.FaceRecognizerSF.create(
        model=str(fr_model_path),
        config="",
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
    
    # list of recognized faces
    # if no face is recognized, return None for that face
    if faces is None:
        return None
    recognized_faces = len(faces)*[None]
    

    for idx, face in enumerate(faces):
        # get features from face recognizer
        features = get_features(img, face[:-1], face_recognizer)

        # compare features with known faces
        for name, known_features in known_faces_dict.items():
            distance = face_recognizer.match(features, known_features, 0)
            print(distance)
            if distance > fr_model_threshold:
                recognized_faces[idx] = name

    return recognized_faces

def detect_faces(frame, w, h):
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # initialize face detector
    face_detector = cv2.FaceDetectorYN.create(
        model=str(fd_model_path),
        config="",
        input_size=(w, h),
        score_threshold=fd_model_threshold
    )

    _, faces = face_detector.detect(img)
    
    visualize(frame, faces)
    return faces

def add_known_faces():
    # Ask user to enter name of the person
    name = input("Enter the name of the person in the image (press spacebar to capture image): ")
    
    # capture image from webcam
    video_capture = cv2.VideoCapture(0)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    captured_frame = None
    
    while True:
        result, frame = video_capture.read()
        if result is False:
            break
        detect_faces(frame, w, h)
        cv2.imshow('Image Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            captured_frame = frame
            break
    video_capture.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        faces = detect_faces(captured_frame, w, h)
    else:
        raise ValueError("No image captured")


    # if more than one faces, throw an error saying only one face is allowed
    if faces is not None:
        if len(faces) > 1:
            raise ValueError("Only one face is allowed")
        # else:
        #     face = faces[0]
        #     coords = face[:-1].astype(np.int32)
            # face_image = captured_frame[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            # cv2.imshow('Add Known Face', face_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    else:
        raise ValueError("No face detected")
    
    # add face feature to known_faces_dict
    known_faces_dict[name] = get_features(captured_frame, faces[0][:-1])


def main():
    # load known faces
    add_known_faces()

    video_capture = cv2.VideoCapture(0)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tm = cv2.TickMeter()
    
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        
        tm.start()
        faces = detect_faces(video_frame, w, h)
        tm.stop()
        recognized_faces = recognize_faces(video_frame, faces)
        visualize_recognized_faces(video_frame, faces, recognized_faces, tm.getFPS())
        cv2.imshow('Face Recognition', video_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()        

if __name__ == "__main__":
    main()
