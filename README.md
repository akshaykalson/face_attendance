Face Recognition Attendance System
This Python project utilizes the OpenCV and face_recognition libraries to create an attendance system using facial recognition. It captures images from a webcam, compares them with pre-existing images of known individuals, and marks their attendance in a CSV file.

How It Works
Image Loading: Images of known individuals are loaded from the specified directory (ImagesAttendance).
Encoding: The facial encodings of these images are computed using the face_recognition library.
Webcam Capture: The project accesses the webcam using OpenCV's cv2.VideoCapture method.
Face Detection: It detects faces in the webcam frames using face_recognition.face_locations.
Face Encoding: Encodes the detected faces using face_recognition.face_encodings.
Matching: Compares the encodings of detected faces with the known encodings to identify matches.
Attendance Marking: Marks the attendance of recognized individuals in a CSV file (Attendance.csv) along with the timestamp.
Setup Instructions
Install Dependencies: Ensure you have the required Python libraries installed (cv2, numpy, face_recognition).

Directory Structure: Organize your images of known individuals in the ImagesAttendance directory.

Run the Script: Execute the Python script (AttendanceSystem.py) to start the attendance system.

Requirements
Python 3.x
OpenCV (cv2)
NumPy
face_recognition
