# ASL Hand Landmark Data Collection

This project is used to collect ASL hand landmark data with a webcam using MediaPipe Hand Landmarker and OpenCV.

The program guides the user through ASL letters, shows hand landmarks live on screen, and saves landmark coordinates into a CSV file for later model training.

## What this project does

- Opens the webcam
- Detects one hand
- Draws the hand skeleton and landmark points
- Labels each landmark from `0` to `20`
- Guides the user through letters `A-Z`
- Skips `J` and `Z` for now because they involve motion
- Saves `40` samples per letter by default
- Stores normalized landmark coordinates in a CSV file

# How to set up on your device.

Python 3.10 or newer recommended

## 1. Create a virtual environment

Mac / Linux

```shell
python3 -m venv venv
source venv/bin/activate
```

Windows

```shell
python -m venv venv
venv\Scripts\activate
```

## 2. Install Dependancies

```shell
pip install -r requirements.txt
```
