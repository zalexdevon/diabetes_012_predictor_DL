from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from Mylib import myfuncs
import os
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import os


class CustomisedModelCheckpoint(tf.keras.callbacks.Callback):
    SCORINGS_PREFER_MININUM = ["loss", "mse", "mae"]
    SCORINGS_PREFER_MAXIMUM = ["accuracy", "roc_auc"]

    def __init__(
        self,
        filepath: str,
        scoring_path: str,
        monitor: str,
        val_scoring_limit_to_save_model: float,
    ):
        """Customized từ class ModelCheckpoint trong tf.keras.callbacks, ở đây thêm logic để save model khi đạt (1) <br>
        (1): val scoring của best model phải tốt hơn val_scoring_limit_to_save_model <br>

        Args:
            filepath (str): đường dẫn lưu model
            scoring_path (str): đường dẫn lưu train, val scoring
            monitor (str): chỉ số
            val_scoring_limit_to_save_model (float): mức cần vượt qua để lưu model
        """
        super().__init__()
        self.filepath = filepath
        self.scoring_path = scoring_path
        self.monitor = monitor
        self.val_scoring_limit_to_save_model = val_scoring_limit_to_save_model

    def on_train_begin(self, logs=None):
        # Nếu thuộc SCORINGS_PREFER_MININUM thì lấy âm đẩy về bài toán tìm giá trị lớn nhất
        self.sign_for_score = None
        if self.monitor in self.SCORINGS_PREFER_MAXIMUM:
            self.sign_for_score = 1
        elif self.monitor in self.SCORINGS_PREFER_MININUM:
            self.sign_for_score = -1
        else:
            raise ValueError(f"Chưa định nghĩa cho {self.monitor}")

        self.train_scorings = []
        self.val_scorings = []
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_scorings.append(logs.get(self.monitor) * self.sign_for_score)
        self.val_scorings.append(logs.get(f"val_{self.monitor}") * self.sign_for_score)
        self.models.append(self.model)

    def on_train_end(self, logs=None):
        # Tìm model ứng với val scoring tốt nhất
        index_best_model = np.argmax(self.val_scorings)
        best_val_scoring = self.val_scorings[index_best_model]
        best_train_scoring = self.train_scorings[index_best_model]
        best_model = self.models[index_best_model]

        # Trước đó lấy âm -> lấy trị tuyệt đối
        best_val_scoring = np.abs(best_val_scoring)
        best_train_scoring = np.abs(best_train_scoring)

        print(f"Model tốt nhất ứng với epoch = {index_best_model + 1}")

        # Lưu kết quả model
        myfuncs.save_python_object(
            self.scoring_path, (best_train_scoring, best_val_scoring)
        )

        # Lưu model
        self.save_model(best_val_scoring, best_model)

    def save_model(self, best_val_scoring, best_model):
        do_allow_to_save_model = self.is_val_scoring_better_than_target_scoring(
            best_val_scoring
        )
        if do_allow_to_save_model:
            best_model.save(self.filepath)

    def is_val_scoring_better_than_target_scoring(self, val_scoring):
        if self.monitor in self.SCORINGS_PREFER_MAXIMUM:
            return val_scoring > self.val_scoring_limit_to_save_model
        if self.monitor in self.SCORINGS_PREFER_MININUM:
            return val_scoring < self.val_scoring_limit_to_save_model

        raise ValueError(f"Chưa định nghĩa cho {self.monitor}")
