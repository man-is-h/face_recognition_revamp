import cv2
import numpy as np
from pathlib import Path

# Make path system-independent
fd_model_path = Path(__file__).parent.parent / "datasources" / "face_detection_model" / "face_detection_yunet_2023mar.onnx"
fd_model_threshold = 0.5

fr_model_path = Path(__file__).parent.parent / "datasources" / "face_recognition_model" / "face_recognition_sface_2021dec.onnx"
fr_model_threshold = 0.5

class FaceRecognition:
    def __init__(self):
        self.known_faces_dict = {}
        self.face_detector = cv2.FaceDetectorYN.create(
            model=str(fd_model_path),
            config="",
            input_size=(640, 480),
            score_threshold=fd_model_threshold,
        )
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            model=str(fr_model_path),
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def visualize(self, frame, faces, thickness=2):
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
            for _, face in enumerate(faces):
                coords = face[:-1].astype(np.int32)
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
    
    def visualize_recognized_faces(self, frame, faces, recognized_faces, fps):
        # 14: face score
        if faces is not None:
            for idx, face in enumerate(faces):
                coords = face[:-1].astype(np.int32)
                if recognized_faces[idx] is not None:
                    cv2.putText(frame, recognized_faces[idx].upper(), (coords[0], coords[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def capture_image(self, num_faces, namesArray):
        # capture image from webcam
        video_capture = cv2.VideoCapture(0)
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        captured_count = 0
        facesArray = []
        framesArray = []
        
        while True:
            result, frame = video_capture.read()
            if result is False:
                break
            frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
            faces = self.detect_faces(frame, w, h)
            cv2.putText(frame, 'Capturing image for '+namesArray[captured_count], (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Capture Image', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                if len(faces) != 1:
                    raise ValueError("Only one face is allowed and required")
                captured_count += 1
                facesArray.append(faces[0][:-1])
                framesArray.append(frame)
                if captured_count >= num_faces:
                    break
        video_capture.release()
        cv2.destroyAllWindows()

        return framesArray, facesArray

    def add_known_faces(self):
        # Ask the user to enter the number of faces to add
        num_faces = int(input("Enter the number of faces to add: "))
        namesArray = []
        for i in range(num_faces):
            # Ask user to enter name of the person
            name = input(f"Enter name of the person {i+1} (press spacebar to capture image): ")
            namesArray.append(name)
        frames, faces = self.capture_image(num_faces, namesArray)
        for i in range(num_faces):
            # add face feature to known_faces_dict
            self.known_faces_dict[namesArray[i]] = self.get_features(frames[i], faces[i][:-1])
    
    def recognize_faces(self, frame, faces):
        # list of recognized faces
        # if no face is recognized, return None for that face
        if faces is None:
            return None
        recognized_faces = len(faces)*[None]
        

        for idx, face in enumerate(faces):
            # get features from face recognizer
            features = self.get_features(frame, face[:-1])

            # compare features with known faces
            for name, known_features in self.known_faces_dict.items():
                similarity = self.face_recognizer.match(features, known_features, 0)
                # print(distance)
                if similarity > fr_model_threshold:
                    recognized_faces[idx] = name
                    break;

        return recognized_faces
    
    def detect_faces(self, frame, w, h):
        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        _, faces = self.face_detector.detect(img)
        self.visualize(frame, faces)
        return faces
    
    def get_features(self, img, face):
        #  align crop face
        face = self.face_recognizer.alignCrop(img, face)
        # get features from face recognizer
        return self.face_recognizer.feature(face)

    def start_recognition(self):
        # Ask user to enter names of the persons to be recognized
        self.add_known_faces()

        # start video capture
        video_capture = cv2.VideoCapture(0)
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        tm = cv2.TickMeter()
        
        while True:
            result, frame = video_capture.read()
            if result is False:
                break
            frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
            tm.start()
            faces = self.detect_faces(frame, w, h)
            recognized_faces = self.recognize_faces(frame, faces)
            tm.stop()
            self.visualize_recognized_faces(frame, faces, recognized_faces, tm.getFPS())
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        
def main():
    fr = FaceRecognition()
    fr.start_recognition()

if __name__ == "__main__":
    main()
