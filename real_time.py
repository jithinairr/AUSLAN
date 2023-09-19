import cv2
import numpy as np
from mediapipe_holistic import MediaPipeHolistic
from data_collection import DataCollector
from keras.models import load_model


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        prob_scalar = float(prob)  # Ensure prob is a scalar
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob_scalar * 300), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


def initialize_realtime_detection(actions, model, threshold=0.5):
    sequence = []
    sentence = []
    predictions = []
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

    cap = cv2.VideoCapture(0)

    model = load_model('signs.h5')

    # Set up the MediaPipe model
    holistic = MediaPipeHolistic()
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = holistic.process_frame(frame)
        holistic.draw_landmarks(image, results)

        keypoints = DataCollector.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-35:]

        if len(sequence) == 35:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)  # Visualize probabilities using prob_viz

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
