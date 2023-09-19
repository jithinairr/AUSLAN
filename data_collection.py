import cv2
import os
import numpy as np
from mediapipe_holistic import MediaPipeHolistic


class DataCollector:
    def __init__(self, data_path, actions, no_sequences, sequence_length, start_folder, holistic_instance):
        self.data_path = data_path
        self.actions = actions
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.start_folder = start_folder
        self.holistic_instance = holistic_instance

    def setup_data_folders(self):
        for action in self.actions:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(self.data_path, action, str(sequence)))
                except:
                    pass

    @staticmethod
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
            33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
            468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        # Create an instance of MediaPipeHolistic
        holistic = MediaPipeHolistic()
        # Loop through actions
        for action in self.actions:
            # Loop through sequences aka videos
            for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
                # Create the directory for this action and sequence if it doesn't exist
                sequence_dir = os.path.join(self.data_path, action, str(sequence))
                if not os.path.exists(sequence_dir):
                    os.makedirs(sequence_dir)
                # Loop through video length aka sequence length
                for frame_num in range(self.sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    image, results = holistic.process_frame(frame)
                    holistic.draw_landmarks(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = self.extract_keypoints(results)
                    npy_path = os.path.join(self.data_path, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
