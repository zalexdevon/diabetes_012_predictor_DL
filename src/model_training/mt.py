import os
from Mylib import myfuncs, tf_myfuncs
import time
import re
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from Mylib.myclasses import CustomStackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import tensorflow as tf
from src.utils import classes


def load_data(data_transformation_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")
    num_features = myfuncs.load_python_object(
        f"{data_transformation_path}/num_features.pkl"
    )

    return train_ds, val_ds, num_features


def save_models_before_training(model_training_path, model_indices, models):
    for model_index, model in zip(model_indices, models):
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )


def create_model_from_layers(model, num_features):
    input_layer = tf.keras.Input(shape=(num_features,))  # Tạo sẵn layer Input trước
    x = input_layer

    for layer in model:
        x = layer(x)

    # Tạo model thôi, chưa cần compile
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    return model


def create_and_save_models_before_training(
    model_training_path, model_indices, models, num_features
):
    for model_index, model in zip(model_indices, models):
        model_folder_path = f"{model_training_path}/{model_index}"
        myfuncs.create_directories_on_colab([model_folder_path])
        model = create_model_from_layers(model, num_features)

        # Save model
        model.save(f"{model_folder_path}/model.keras")


def create_callbacks(
    callbacks, model_folder_path, scoring_path, scoring, val_scoring_limit_to_save_model
):
    callbacks = [tf_myfuncs.copy_one_callback(callback) for callback in callbacks]

    callbacks = [
        classes.CustomisedModelCheckpoint(
            filepath=f"{model_folder_path}/fitted_model.keras",
            scoring_path=scoring_path,
            monitor=scoring,
            val_scoring_limit_to_save_model=val_scoring_limit_to_save_model,
        )
    ] + callbacks

    return callbacks


def train_and_save_models(
    model_training_path,
    model_indices,
    train_ds,
    val_ds,
    epochs,
    callbacks,
    model_name,
    scoring,
    optimizer,
    loss,
    val_scoring_limit_to_save_model,
):
    tf.config.run_functions_eagerly(True)  # Bật eager execution
    tf.data.experimental.enable_debug_mode()  # Bật chế độ eager cho tf.data

    logging_message = ""
    for model_index in model_indices:
        # Load model
        model_folder_path = f"{model_training_path}/{model_index}"
        model = tf.keras.models.load_model(f"{model_folder_path}/model.keras")
        os.remove(f"{model_folder_path}/model.keras")

        # Create optimizer cho model
        model_optimizer = tf_myfuncs.copy_one_optimizer(optimizer)

        # Compile model trước khi training
        model.compile(optimizer=model_optimizer, loss=loss, metrics=[scoring])

        # Create callbacks cho model
        scoring_path = f"{model_folder_path}/scoring.pkl"
        model_callbacks = create_callbacks(
            callbacks,
            model_folder_path,
            scoring_path,
            scoring,
            val_scoring_limit_to_save_model,
        )

        # Train model và lưu model ứng với val_scoring tốt nhất
        full_model_index = f"{model_name}_{model_index}"

        print(f"Train model {full_model_index}")
        start_time = time.time()
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=model_callbacks,
        ).history
        training_time = time.time() - start_time
        num_epochs_before_stopping = f"{len(history.history["loss"])}/ {epochs}" # Số epoch trước khi dừng train model 
        train_scoring, val_scoring = myfuncs.load_python_object(scoring_path)
        os.remove(scoring_path) # Không cần thiết nữa 

        ## In kết quả
        training_result_text = f"Model {full_model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Time: {training_time} (s), Epochs: {num_epochs_before_stopping}\n"
        print(training_result_text)

        # Logging 
        logging_message += training_result_text

        # Lưu kết quả train model
        myfuncs.save_python_object(
            f"{model_folder_path}/result.pkl",
            (full_model_index, train_scoring, val_scoring, training_time, num_epochs_before_stopping),
        )
    
    return logging_message