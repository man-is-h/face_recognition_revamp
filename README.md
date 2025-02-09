Revamping the face recognition project moving from still images to real time video recognition and exploring new deep learning models, apart from ResNet-50.

Approach:
1. Use a pre-trained model to extract facial features from the video frames.
2. Use a deep learning model to classify the facial features and generate a face embedding.
3. Use a similarity metric to compare the face embedding to the face embeddings of the known individuals.
4. If a match is found, render the name of the individual on the video frame.