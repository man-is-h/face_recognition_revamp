# Main module for the face recognition project
# create a skeleton main function to call all other modules

import cv2
import numpy as np
from pathlib import Path

# Make path system-independent
model_path = Path(__file__).parent.parent / "datasources" / "face_detection_classifier" / "face_detection_yunet_2023mar.onnx"
model_threshold = 0.5

def visualize(input, faces, fps, thickness=2):
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
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def detect_faces(frame, w, h):
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    face_detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=model_threshold
    )
    face_detector.setInputSize([w, h])

    tm = cv2.TickMeter()
    tm.start()
    _, faces = face_detector.detect(img)
    tm.stop()
    visualize(frame, faces, tm.getFPS())



def main():
    video_capture = cv2.VideoCapture(0)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        detect_faces(video_frame, w, h)
        cv2.imshow('Face Detection', video_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
