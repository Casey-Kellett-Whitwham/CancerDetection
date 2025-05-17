import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def build_model(train_generator, input_shape=(64, 64, 3), num_classes=26):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(32, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def train_model(model, train_generator, test_generator, epochs=20, batch_size=16):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    return history
