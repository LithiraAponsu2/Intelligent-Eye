# Traffic Violation Detection at Pedestrian Crossing with Computer Vision

## Overview
This project aims to develop a computer vision-based system for detecting traffic violations at pedestrian crossings. The system utilizes machine learning techniques to analyze video feeds from traffic cameras and identify instances where vehicles violate pedestrian right-of-way at crossings. The Project is On going and the Folder Names were the files were created for each version for easiness. Latest Folder contains the most Updated Version.

## Blog
https://sites.google.com/view/lithira-aponsu/home

## Goals
- Detect vehicles approaching pedestrian crossings.
- Identify instances where vehicles fail to yield to pedestrians.
- Generate alerts or notifications for law enforcement or traffic management authorities.

## Dataset
We collected a dataset consisting of video footage from various pedestrian crossings in urban areas. Each video contains labeled instances of pedestrian crossings, vehicles, and pedestrian traffic.

## Methodology
1. **Data Preprocessing**: Extract frames from video footage, annotate pedestrian crossings, vehicles, and pedestrians.
2. **Feature Engineering**: Extract features such as vehicle speed, distance from crossing, pedestrian density, etc.
3. **Model Selection**: Experiment with various machine learning models such as YOLO (You Only Look Once), SSD (Single Shot Multibox Detector), or Faster R-CNN (Region-based Convolutional Neural Networks).
4. **Training**: Train the selected model on the annotated dataset.
5. **Evaluation**: Evaluate the model's performance using metrics such as precision, recall, and F1-score.
6. **Deployment**: Integrate the trained model into a real-time system capable of processing video feeds from traffic cameras.

## Results
- Achieved a detection accuracy of over 90% on the test dataset.
- Successfully identified and flagged instances of traffic violations in real-time.

## Future Improvements
- Incorporate additional features such as vehicle type recognition, pedestrian behavior analysis, etc.
- Fine-tune the model to reduce false positives and negatives.
- Explore the integration of the system with traffic management infrastructure for automated enforcement.

## Conclusion
The developed system demonstrates the potential of computer vision and machine learning in enhancing traffic safety at pedestrian crossings. By accurately detecting and penalizing traffic violations, we aim to contribute to the creation of safer urban environments for pedestrians.
