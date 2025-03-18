import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_pattern):
    all_files = glob.glob(file_pattern)
    processed_data = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        X_processed, y_processed = preprocess_data(df)
        if X_processed.size > 0:
            processed_data.append((X_processed, y_processed))

    X_combined = np.concatenate([data[0] for data in processed_data], axis=0) if processed_data else np.array([])
    y_combined = np.concatenate([data[1] for data in processed_data], axis=0) if processed_data else np.array([])

    return X_combined, y_combined

def preprocess_data(frame):
    frame['time'] = pd.to_datetime(frame['time'])
    frame['day'] = frame['time'].dt.day
    frame['month'] = frame['time'].dt.month
    frame['time_in_hour'] = frame['time'].dt.hour
    frame['time_in_minute'] = frame['time'].dt.minute
    frame['time_in_seconds'] = frame['time'].dt.second
    frame['plate'] = frame['plate'].astype('int64')

    unique_plates = frame['plate'].unique()
    scaler = StandardScaler()

    X_reshaped, y_reshaped = [], []

    for plate in unique_plates:
        plate_frame = frame[frame['plate'] == plate]
        X_plate = plate_frame[['longitude', 'latitude', 'status', 'day', 'month', 'time_in_hour', 'time_in_minute', 'time_in_seconds']].values

        if len(X_plate) > 0:
            X_scaled = scaler.fit_transform(X_plate)

            num_chunks = len(X_scaled) // 100
            for i in range(num_chunks):
                chunk = X_scaled[i*100:(i+1)*100]
                X_reshaped.append(chunk)
                y_reshaped.append(plate)

    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)

    return X_reshaped, y_reshaped
