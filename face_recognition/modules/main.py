import cv2
import numpy as np
from pathlib import Path

# Make path system-independent
fd_model_path = Path(__file__).parent.parent / "datasources" / "face_detection_model" / "face_detection_yunet_2023mar.onnx"
fd_model_threshold = 0.5

fr_model_path = Path(__file__).parent.parent / "datasources" / "face_recognition_model" / "face_recognition_sface_2021dec.onnx"
fr_model_threshold = 0.5

class FaceRecognition:
    def __init__(self, input_size=(640, 480)):
        """Initialize face recognition system with optimized settings.
        
        Args:
            input_size (tuple): Width and height for processing. Using a fixed size
                              improves performance. Default (640, 480).
        """
        self.known_faces_dict = {}
        self.input_size = input_size
        # Pre-allocate frame buffer
        self.frame_buffer = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        
        # Configure face detector with OpenCV DNN optimizations
        self.face_detector = cv2.FaceDetectorYN.create(
            model=str(fd_model_path),
            config="",
            input_size=input_size,
            score_threshold=fd_model_threshold,
            backend_id=cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_TARGET_CPU
        )
        
        # Configure face recognizer with optimizations
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            model=str(fr_model_path),
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_TARGET_CPU
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
        """Visualize recognized faces, supporting both single and multiple matches per face."""
        if faces is not None:
            for idx, face in enumerate(faces):
                coords = face[:-1].astype(np.int32)
                if isinstance(recognized_faces[idx], list) and recognized_faces[idx]:
                    # Multiple matches mode - show top 3 matches with scores
                    for i, (name, score) in enumerate(recognized_faces[idx][:3]):
                        text = f"{name.upper()} ({score:.2f})"
                        y_offset = -4 - (i * 25)  # Stack names vertically
                        cv2.putText(frame, text, (coords[0], coords[1]+y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif recognized_faces[idx]:
                    # Single match mode - show just the name
                    cv2.putText(frame, recognized_faces[idx].upper(), 
                              (coords[0], coords[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
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
            frame = cv2.flip(frame, 1)
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
    
    def recognize_faces(self, frame, faces, return_all_matches=False):
        """Recognize faces with optimized batch processing.
        
        Args:
            frame: Input frame containing faces
            faces: Detected face coordinates
            return_all_matches: If True, returns list of all matches above threshold
                              for each face. If False, returns only best match.
        
        Returns:
            If return_all_matches=False: List of best matches (one per face)
            If return_all_matches=True: List of lists containing all matches per face
                                      with similarities above threshold
        """
        if faces is None:
            return None
            
        recognized_faces = [[] if return_all_matches else None] * len(faces)
        if not self.known_faces_dict:
            return recognized_faces
            
        # Extract features for all faces in one batch
        features_list = [self.get_features(frame, face[:-1]) for face in faces]
        
        # Prepare known faces array for vectorized comparison
        known_names = list(self.known_faces_dict.keys())
        known_features = np.array([self.known_faces_dict[name] for name in known_names])
        
        # Compare each detected face against all known faces efficiently
        for idx, features in enumerate(features_list):
            # Compute similarities with all known faces at once
            similarities = np.array([self.face_recognizer.match(features, kf, 0) for kf in known_features])
            
            if return_all_matches:
                # Get all matches above threshold with their scores
                matches = [(name, score) for name, score in zip(known_names, similarities) 
                          if score > fr_model_threshold]
                # Sort by similarity score in descending order
                matches.sort(key=lambda x: x[1], reverse=True)
                recognized_faces[idx] = matches
            else:
                # Return only the best match if above threshold
                max_similarity_idx = np.argmax(similarities)
                if similarities[max_similarity_idx] > fr_model_threshold:
                    recognized_faces[idx] = known_names[max_similarity_idx]
                
        return recognized_faces
    
    def detect_faces(self, frame, w, h):
        """Detect faces with optimized preprocessing."""
        # Reuse pre-allocated buffer for color conversion
        cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR, dst=self.frame_buffer)
        _, faces = self.face_detector.detect(self.frame_buffer)
        self.visualize(frame, faces)
        return faces
    
    def get_features(self, img, face):
        #  align crop face
        face = self.face_recognizer.alignCrop(img, face)
        # get features from face recognizer
        return self.face_recognizer.feature(face)

    def start_recognition(self, skip_frames=1):
        """Start face recognition with performance optimizations.
        
        Args:
            skip_frames (int): Process every nth frame to reduce CPU load.
        """
        self.add_known_faces()

        # Configure video capture for performance
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        tm = cv2.TickMeter()
        frame_count = 0
        last_faces = None
        last_recognized = None
        
        try:
            while True:
                result, frame = video_capture.read()
                if not result:
                    break
                    
                frame_count += 1
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, self.input_size, dst=self.frame_buffer)
                
                # Process only every nth frame for detection/recognition
                if frame_count % skip_frames == 0:
                    tm.start()
                    faces = self.detect_faces(frame, w, h)
                    if faces is not None and len(faces) > 0:
                        # Get all matches above threshold with their scores
                        recognized_faces = self.recognize_faces(frame, faces, return_all_matches=True)
                        last_faces, last_recognized = faces, recognized_faces
                    tm.stop()
                
                # Always show last detection results
                if last_faces is not None:
                    self.visualize_recognized_faces(frame, last_faces, last_recognized, tm.getFPS())
                
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
        video_capture.release()
        cv2.destroyAllWindows()
        
def main():
    fr = FaceRecognition()
    fr.start_recognition()

if __name__ == "__main__":
    main()
