# Intelligent Video Surveillance Using Deep Learning

This project implements an intelligent video surveillance system using deep learning techniques. The system preprocesses video frames, trains a spatial temporal autoencoder model to detect abnormal events in video surveillance.

## Features

- *Upload Video Frames Dataset*: Upload a directory of video frames for preprocessing.
- *Dataset Preprocessing*: Preprocess the uploaded video frames for model training.
- *Train Spatial Temporal AutoEncoder Model*: Train the model using the preprocessed dataset.
- *Test Video Surveillance*: Test the trained model on a new video to detect abnormal events.
- *Exit*: Exit the application.

## Requirements

- Python 3.x
- Tkinter
- Keras
- NumPy
- OpenCV
- Imutils
- PIL

## Installation

1. *Clone the repository*:
    bash
    git clone https://github.com/yourusername/intelligent-video-surveillance.git
    cd intelligent-video-surveillance
    

2. *Install the required libraries*:
    bash
    pip install numpy keras opencv-python imutils pillow
    

## Usage

1. *Run the application*:
    bash
    python app.py
    

2. *Upload Video Frames Dataset*:
    - Click on the "Upload Video Frames Dataset" button.
    - Select the directory containing the video frames.

3. *Dataset Preprocessing*:
    - Click on the "Dataset Preprocessing" button.
    - The application will preprocess the images and display the number of images found in the dataset.

4. *Train Spatial Temporal AutoEncoder Model*:
    - Click on the "Train Spatial Temporal AutoEncoder Model" button.
    - The application will train the model and save it in the model directory.

5. *Test Video Surveillance*:
    - Click on the "Test Video Surveillance" button.
    - Select the video file to test.
    - The application will display the video and detect abnormal events.

6. *Exit*:
    - Click on the "Exit" button to close the application.

## Code Overview

- *app.py*: Main application file that contains the GUI and functionality for video surveillance.
- *model*: Directory to save the trained model.
- *Dataset*: Directory to store the uploaded video frames.

### Main Functions

- readImages(path): Reads and preprocesses an image from the given path.
- upload(): Opens a file dialog to select and upload a directory of video frames.
- datasetPreprocess(): Preprocesses the uploaded video frames for model training.
- meanLoss(image1, image2): Calculates the mean loss between two images.
- trainModel(): Trains the spatial temporal autoencoder model and saves it.
- abnormalDetection(): Tests the trained model on a new video to detect abnormal events.
- close(): Closes the application.

### GUI Components

- *Title*: Displays the title of the application.
- *Text Box*: Displays logs and messages.
- *Buttons*: Provides functionality to upload dataset, preprocess dataset, train model, test surveillance, and exit.

## Notes

- Ensure the Dataset directory contains video frames before preprocessing and training the model.
- The model will be saved in the model directory after training.
- Abnormal events in the video will be highlighted with a red text "Abnormal Event".

## License

This project is licensed under the MIT License.