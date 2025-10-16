Face Recognition using OpenCV and Deep Learning

This is an updated and improved version of the previous face recognition project - https://github.com/man-is-h/face_model

Current max FPS achieved is 20 FPS using the YuNet mode for face detection + Sface model for face recognition.
___

Specs:

- FPS: 60
- CPU: Macbook Air M4 (8 core)
- RAM: 16GB
___

- FPS: 20
- CPU: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 4 Core(s)
- RAM: 8GB
___

Sequence :
1. Capture images of the faces of the individuals to be recognized.
2. Use SFace recognizer to extract features from the face.
3. Use a similarity metric to compare the face embedding to the face embeddings of the known individuals.
4. If a match is found, render the name of the individual on the video frame.

How to run:

1. Clone this repo
2. Create a virtual env and install dependencies specified in requirements.txt
3. Run using python modules/main.py