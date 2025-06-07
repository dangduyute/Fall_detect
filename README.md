# Fall Detection System using ESP32 & Machine Learning:
This is a real-time fall detection system use the MPU6050 motion sensor and a deep learning model deployed on the ESP32 platform with Firebase integration, cambine with sensor including MLX90614 and MAX30100 to easure body temperature, heart rate, and blood oxygen (SpO₂) data.

Features:
- Collects motion data from the MPU6050 sensor (accelerometer and gyroscope).
- Collects temperature, heart rate, and blood oxygen data and upload to Firebase. 
- Classifies user activities (walking, sitting, falling, etc.).
- Sends fall alerts to Firebase Realtime Database.
- Real-time health data monitoring interface.
- Optional integration with heart rate (MAX30100) and water level sensors.
