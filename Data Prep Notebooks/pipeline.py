import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

def loadprocesseddata(csv_path):
    df = loaddata(csv_path)
    train_df, test_df, val_df = split_df(df)
    train_generator, test_generator, val_generator = create_generators(train_df, test_df, val_df)
    return train_generator, test_generator, val_generator


def loaddata(csv_path):
    df = pd.read_csv(csv_path)
    return df


def split_df(df):

    train_df, test_val_df = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['label'], random_state=42)
    return train_df, test_df, val_df


def create_generators(train_df, test_df, val_df, batch_size=32, img_size=(64, 64)):

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    val_generator = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, test_generator, val_generator


def prepare_for_train(dftrain, dftest):
    train_generator, test_generator, val_generator = create_generators(dftrain, dftest)
    return train_generator, test_generator, val_generator
