import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

# Set random seed for TensorFlow
tf.random.set_seed(42)

# Set random seed for NumPy
np.random.seed(42)


def create_model(input_shape, num_actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    return model


def train_and_save_model(X_train, y_train, actions):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    input_shape = X_train.shape[1:]  # Input shape based on your data

    model = create_model(input_shape, len(actions))

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)  # Adjust learning rate as needed
    early_stopping = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001, patience=10, restore_best_weights=True)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Experiment with batch size and number of epochs for hyperparameter tuning
    model.fit(X_train, y_train, epochs=1000, batch_size=32, callbacks=[tb_callback, early_stopping], shuffle=False)

    model.save('signs.h5')


def load_and_predict_model(model_path, X_test):
    model = load_model(model_path)
    predictions = model.predict(X_test)

    return predictions
