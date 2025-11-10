# Handwritten Digit Recognition (MNIST + CNN + OpenCV)

This project performs real-time handwritten digit recognition using a webcam. 
A Convolutional Neural Network (CNN) trained on the MNIST dataset predicts digits 
(0–9) from user handwriting inside a bounding box captured through OpenCV.

## Features
- Real-time camera-based digit detection
- Pre-trained MNIST CNN model
- Shows predicted digit on screen
- Displays Test Loss & Accuracy

## Technologies Used
- Python
- TensorFlow / Keras (CNN Model)
- OpenCV (Camera Input & Display)
- NumPy

## How to Run
pip install tensorflow opencv-python numpy
python camera_predict.py


Press `q` to exit the camera window.

## Model Used
- Dataset: MNIST (70,000 handwritten digit images)
- Input Size: 28×28 grayscale
- Model Type: Convolutional Neural Network (CNN)

## Author
Your Name
