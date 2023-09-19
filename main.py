import os
import numpy as np
import cv2
from camera import Camera
from mediapipe_holistic import MediaPipeHolistic
from data_collection import DataCollector
from preprocess_data import load_data, split_data
from model import train_and_save_model, load_and_predict_model
from real_time import initialize_realtime_detection

DATA_PATH = os.path.join('AUSLAN_Data')
actions = np.array(['hello', 'how are you', 'thank you'])
no_sequences = 100
sequence_length = 35
start_folder = 0


def main():
    camera = Camera()
    holistic = MediaPipeHolistic()

    # Check if the trained model exists
    if os.path.exists('signs.h5'):
        print("Loading the pre-trained model...")
        model = 'signs.h5'
    else:
        print("No pre-trained model found. Training a new model...")
        # Check if the data folder already contains data
        data_exists = all(os.path.exists(os.path.join(DATA_PATH, action, str(sequence), "0.npy"))
                          for action in actions
                          for sequence in range(no_sequences))
        if not data_exists:
            # Data collection is needed
            data_collector = DataCollector(DATA_PATH, actions, no_sequences, sequence_length, start_folder, holistic)
            data_collector.setup_data_folders()

            while camera.cap.isOpened():
                frame = camera.read_frame()
                image, results = holistic.process_frame(frame)
                holistic.draw_landmarks(image, results)
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            data_collector.collect_data()

        # Load and preprocess the data
        X, y = load_data(DATA_PATH, actions, no_sequences, sequence_length)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30)

        camera.release()
        cv2.destroyAllWindows()

        model = 'signs.h5'
        # Train the model
        train_and_save_model(X_train, y_train, actions)

    # Load and preprocess the data for testing
    X, y = load_data(DATA_PATH, actions, no_sequences, sequence_length)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30)

    # Make predictions with the loaded model
    loaded_model = load_and_predict_model(model, X_test)

    # Call initialize_realtime_detection with appropriate arguments
    initialize_realtime_detection(actions, loaded_model, threshold=0.7)


if __name__ == "__main__":
    main()
