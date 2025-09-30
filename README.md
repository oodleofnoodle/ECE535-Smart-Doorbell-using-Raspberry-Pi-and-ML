# ECE535-Smart-Doorbell-using-Raspberry-Pi-and-ML

# Motivation

Much advancing research has been dedicated to creating and deploying light-weight machine learning models on embedded systems for real-world applications. Typically, embedded applications focused on real-world settings rely on cloud computing where a small embedded device sends information to a larger server that runs higher level programs, such as facial recognition programs, from the data provided from the embedded device. Bringing this computation to the edge can improve system latency and lighten the load on the cloud infrastructure. One such application is a Smart Doorbell that could create a safe and convenient system that can identify visitors and provide useful alerts without relying on this heavy cloud infrastructure. 

# Design Goals

The goal of this project is to deploy a light weight machine learning module on an embedded device (Raspberry Pi with a camera module) that can detect and classify whether certain people in front of the system are recognized or unknown. This light weight model should be able to effectively classify the visitors with high accuracy (close to 70% as described in MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications) while also having low latency and small storage use. The objective is to effectively deploy a light weight model that can be customized for embedded applications based on size, latency, and accuracy constraints via alterable hyper parameters. 


# Deliverables
• Learn how to deploy ML models on Raspberry Pi
• Implement a lightweight face detection or person detection model using TensorFlow Lite.
• Build a simple system that, when someone appears at the door, captures an image and classify
• The final output should be a code snippet and demonstration running inference directly on Raspberry Pi
• Optional: Add an alert mechanism (e.g., send a notification to a phone or log it in a file)
• Fun feature!: Can it detect a delivery person? E.g., pizza or package delivery worker?
• The final output should be a code snippet and demonstration running inference directly on
Raspberry Pi

# System Blocks 
In order to implement the light-weight model on a Raspberry Pi using TensorFlow Lite, we will use the following system block:

Camera Input --> Preprocessing --> TensorFlow Lite Model Inference --> Postprocessing --> Inference Output --> Model Update

Each stage of this system block shows a different process that must be done in order to accomplish the deliverables of the project. Below describes each block and the actions that should take place during those blocks

The camera input block is the beginning stage of the system. Here the camera module on the Raspberry pi will capture images or video snippets that will then be sent through a trained model for facial or object recognition in the later blocks.

The preprocessing block then prepares the data from the image for inference. This could be scaling the image in such a way to that is expected for the model such as a 224x224 image or a 192x192 image as used in the MobileNet paper. This would then involve scaling the pixel values of the image such as normalizing the RGB values of each pixel to match a 0 to 1 scale. 

The TensorFlow Lite Model Block then takes the pretrained TensorFlow lite model and takes the preprocessed image as an input and then produces an output feature vector that is then used to characterize the input for facial recognition. The TensorFlow Lite implementation based off the MobileNet paper optimizing on available space and latency requirements of the door bell modifying the width and resolution hyper-parameters for this application. 

The postprocessing block then takes the resulting output vector and characterizes the image based on this vector. This block would essentially interpret the vector and classify it based on the face data set for the project. This could be comparing it to a set of trusted data sets that describe different people's faces or compare the results to trusted businesses like delivery person outfits.

The inference output block then takes the information learned about the image from the postprocessing block and completes an action based on the face / body / outfit of the person (such as sending phone notifications or triggering alarms). 

Finally the model update block describes the process of re-optimizing the TFLite model for the application. Actions in this block could be keeping track of power and latency of the model, optimizing on which model version with different hyper-parameters are used in different situations based on results, or possibly updating the TFLite model hyperparameters itself. 

# Hardware Requirements
Raspberry Pi 
Pi Camera Module
MicroSD

# Software Requirements 
TensorFlow Lite
Google Colab


# Project Timeline
Sep 28 : Project Proposal Submission; set up GitHub Repository; install TensorFlow and TensorFlow Lite

Oct 1: Study the MobileNet paper in depth focusing on depthwise separable convolutions and hyper parameters; review how to implement TensorFlow of MobileNet; look into what it means to have a TensorFlow Lite conversion for the model

Oct 8: Choose a dataset for training the model and download the data sets; look into how to preprocess the dataset for the model we want to use; decide on Raspberry pi device and camera module (maybe pick up device from Professor Fatima if already provided)

Oct 16: Do a preliminary baseline training possibly with the MobileNet model from the background paper; Write down and look at the accuracy, latency, and size of the model; download Raspberry Pi OS and run small command to familiarize with Raspberry Pi

Oct 23: Convert the preliminary model into a TensorFlow LIte model; run the converted model on the desktop to check on the accuracy of the model as compared to the preliminary one; integrate the camera module into the Raspberry Pi to take a picture
Oct 30: Adjust the width and resolution parameters to optimize the TensorFlow Lite model for the embedded application; look at the trade-offs of the model as compared to the last original converted model; put the model on the Raspberry pi for its relative size constraint

Nov 6: Implement the rest of the system blocks other than the model update and run a test of the system; begin debugging while measuring the latency and accuracy

Nov 13: Continue debugging the system and implement changes; begin looking into post processing actions such as sending a notification or having some physical action and UI for the application

Nov 20: Continue implementing postprocessing block; analyze the performance of the model and begin implementing the model update system block 

Nov 27: Continue implementing the model update system block; finalize the post processing block and begin to look into other optional applications

Dec 4: Finalize system; begin drafting presentation and project report

Dec 11: Finalize project report and presentation documents



# References
Howard, A. et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (lightweight CNN background). (https://arxiv.org/abs/1704.04861)

https://github.com/lyk125/MobileNet-1 

TensorFlow Lite documentation



# Team Member Responsibilities and Lead Roles 
Jessie Wang - Lead: Setup and Writing and Machine Learning
Responsibilities: 
Setup Github repository
Setup and configure selected Raspberry Pi, integrate and mount selected Pi camera module
Document steps, implementations, challenges, problems and solutions, research

Alex Andoni - Lead: Software and Algorithm Design 
Responsibilities:
Install and understand how to use TensorFlow Lite environments and MobileNet
Choose and train datasets
Convert trained models to TensorFlow Lite
Optimize models for embedded usage 

Lucas Crawshaw - Lead: Research, Networking, and Embedded
Responsibilities: 
Research and implement post-processing actions
Investigate lightweight ML models for Raspberry Pi
Configuring the Pi’s Wi-Fi or Ethernet connection

Vinayak Kapur - Lead: ML Optimization, Phone Alerts, and Detection Responsibilities: Develop and tune models for detection and work on dynamic phone alerts and classifying delivery workers/suspicious activity. Assist in model training and hyperparameter tuning. 



