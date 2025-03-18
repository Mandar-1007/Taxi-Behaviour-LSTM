# Deep Learning for Driver Recognition from GPS Sequences

This project aims to classify taxi drivers based on their GPS trajectory data. By analyzing the movement patterns of taxis, we can identify different drivers based on how they navigate the city. This can be useful for fleet management, fraud detection, and personalized driver performance insights.

The model has been optimized to achieve a testing accuracy of 85.26%, demonstrating its effectiveness in identifying drivers based on their driving behavior.

# Introduction

Taxi companies and ride-sharing platforms generate a large amount of GPS data. Each driver has a unique driving pattern based on their route choices, driving speed, and timing. This project uses machine learning to classify drivers based on these movement patterns. Instead of manually analyzing each driver’s movement, we automate the process using AI, making predictions based on their past trips.

# How it works 

The model learns from past driving patterns and predicts which driver a trip belongs to.
- It takes GPS coordinates and timestamps from taxi trips.
- The model recognizes patterns in movement over time.
- After training, it can classify a trip to a specific driver.
Unlike traditional methods that look at static data (e.g., average speed), this model analyzes sequences of data over time, improving accuracy.

In this project, we to finish a sequence classification task using deep learning. A trajectory data set with five taxi drivers' daily driving trajectories in 6 months is used. The primary objective is to predict which driver each 100-step sub-trajectory, extracted from the daily trajectories, belongs to. To evaluate the model, it will be tested on a separate set of data for five additional days (5 CSV files, same format as the training data), using the same preprocessing steps to ensure consistent data handling. This approach ensures consistency in data preparation across training and testing phases, allowing the model to accurately attribute each sub-trajectory to the correct driver.

![image](https://github.com/user-attachments/assets/d3c4b2e0-9eb1-49fc-a2cb-00f4b2dffb3c)



# Dataset

We use GPS data collected from multiple taxi drivers. The dataset consists of multiple CSV files, each containing:
- Plate Number – Unique identifier for the drive
- Longitude & Latitude – The taxi's location at a specific time
- Time – The timestamp of the GPS recording
- Status – Whether the taxi is occupied or available

![image](https://github.com/user-attachments/assets/3226b1f5-aa14-4554-80a3-efa3aacf5a73)

Above is an example of what the data looks like. Each trajectory step is detailed with features such as longitude, latitude, time, and status.

# Data Processing

Raw GPS data isn't immediately usable for training a model. We perform several preprocessing steps:

1) Extracting Time-Based Features:
- Day, month, hour, minute, and second are extracted from timestamps.
2) Standardization:
- All numeric values are normalized so that they have similar ranges.
3) Segmenting Trips:
- Trips are broken into 100-step sequences to provide structured input for learning.

# Model Training

The model is trained using a deep learning architecture that processes sequential GPS data:
- Learns movement patterns over time
- Identifies unique driver behavior based on trajectories

**Model Features**
- LSTM Layer: Captures sequential GPS movements.
- Batch Normalization: Normalizes feature values.
- Dropout Layer: Prevents overfitting.
- Gradient Clipping: Prevents extreme updates.

**KEY FEATURES**

- Handles Sequential Data: Understands patterns in movement over time.
- Learns from Experience: Improves with more data.
- Uses Dropout: Prevents overfitting by making the model more generalizable.
- Gradient Clipping: Ensures stable training by preventing extreme updates.

# Evaluation & Results

**Training Accuracy Progression**

![image](https://github.com/user-attachments/assets/45af4a5c-3772-4947-806c-0fbb8660d5a6)


- The model was trained over 30 epochs, reaching a training accuracy of 88.27%.

**Testing Accuracy**
- The model was evaluated on unseen data and achieved a final testing accuracy of 85.26%.

**Comparison with Other Models**

![image](https://github.com/user-attachments/assets/13ac846c-ec82-4880-9911-e6a55159be7f)

By improving preprocessing and hyperparameters, the current model significantly outperforms earlier versions.

# Environment and Dependencies

This project was developed in Google Colab. To ensure consistency across different environments, install the following required dependencies:

![image](https://github.com/user-attachments/assets/a347ebd0-dbb6-4cdd-8b4c-c2456b1dd242)


# Conclusion

**1) Effectiveness of LSTM for Sequential GPS Data**
- The LSTM-based model successfully classified taxi drivers based on their GPS trajectories, demonstrating the effectiveness of recurrent architectures in handling sequential data.
- Incorporating time-based features (hour, day, etc.) improved the model’s ability to capture driver behavior patterns.

**2) Optimized Training Process**
- The training process was methodically improved through hyperparameter tuning, including dropout rate, learning rate, and the number of epochs.
- Using early stopping prevented overfitting, ensuring that the model generalizes well to unseen data.

**3) Final Performance and Achievements**
- Training Accuracy: ~88.27% (after tuning)
- Testing Accuracy: 85.26% (on unseen test data)
- The model outperformed a baseline fully connected neural network, proving the effectiveness of LSTM and feature engineering.

**4) Challenges Faced & Overcome**
- Avoiding Overfitting: Dropout regularization and batch normalization helped prevent overfitting while improving generalization.
- Optimizing Hyperparameters: A step-by-step tuning approach allowed for small, incremental accuracy improvements without drastic changes.
- Dataset Variability: GPS trajectory data can vary significantly, but standardization and segmentation ensured a fair model comparison.

**5) Potential Future Improvements**
- Adding More Spatial Features: Distance from key locations (e.g., city center, hotspots) could improve predictions.
- Experimenting with More Advanced Architectures: Trying bidirectional LSTM or transformers for further accuracy improvements.
- Larger Dataset & Transfer Learning: Training on a larger, more diverse dataset could boost performance.

This project successfully built an LSTM-based classifier for taxi drivers using GPS data, achieving strong accuracy on real-world trajectory data. By carefully tuning hyperparameters and optimizing training, we improved classification accuracy while avoiding overfitting. Future work can further enhance model performance by incorporating additional spatial-temporal features and more advanced deep learning techniques
