# Smart Wastewater Segregation System

## Executive Abstract
The **Smart Wastewater Segregation System** is an intelligent, automated pipeline designed to optimize water recycling in institutional and industrial environments. Operating on a Raspberry Pi 5 edge-compute node, the system solves the limitations of single-sensor water analysis by employing a **Sensor Fusion Neural Network**. 

It simultaneously acquires chemical data via a custom-calibrated Total Dissolved Solids (TDS) sensor sampling at 128 Hz, and spatial turbidity data via a sequential four-channel (Red, Green, Blue, White) multispectral optical array. The data is processed through a hybrid deep learning model: a MobileNetV2 Convolutional Neural Network (CNN) extracts visual features, while a Dense Multi-Layer Perceptron handles the scalar chemical data. Operating with a strict ~3.0-second batch inference latency, the system dynamically actuates a solenoid valve manifold to segregate the water. By cross-referencing visual contaminants with invisible ionic loads, the system achieves highly credible, real-time wastewater classification.



## Hardware Architecture
* **Core Compute:** Edge processing via Raspberry Pi 5.
* **Chemical Acquisition:** Analog TDS probe interfaced via an I2C-driven ADS1115 ADC, utilizing a calibrated formula for PPM conversion.
* **Optical Acquisition:** Sequential multispectral imaging (RGB LED Array + USB Webcam) to isolate specific dye and turbidity wavelengths.
* **Actuation Logic:** Interrupt-driven flow metering (YF-S201) triggering a 3-way 12V solenoid relay manifold based on dynamic AI inference.
* **AI Model Pipeline:** A hybrid neural network merging a Transfer-Learned MobileNetV2 CNN backbone with a Dense scalar network, fed with robustly scaled data.
* **Telemetry:** Local Flask-based web dashboard providing real-time engineering metrics, system latency, and multi-channel image caching.

## Software Stack
* **Backend Control:** Python 3, GPIO Zero
* **Machine Learning:** TensorFlow/Keras, OpenCV, Scikit-Learn
* **Web Interface:** Flask, Bootstrap 5
