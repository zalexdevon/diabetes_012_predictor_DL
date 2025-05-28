import pandas as pd
from Mylib import myfuncs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BeforeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        #  Đổi tên cột
        rename_dict = {
            "Diabetes_012": "Diabetes_012_target",
            "HighBP": "HighBP_bin",
            "HighChol": "HighChol_bin",
            "CholCheck": "CholCheck_bin",
            "BMI": "BMI_num",
            "Smoker": "Smoker_bin",
            "Stroke": "Stroke_bin",
            "HeartDiseaseorAttack": "HeartDiseaseorAttack_bin",
            "PhysActivity": "PhysActivity_bin",
            "Fruits": "Fruits_bin",
            "Veggies": "Veggies_bin",
            "HvyAlcoholConsump": "HvyAlcoholConsump_bin",
            "AnyHealthcare": "AnyHealthcare_bin",
            "NoDocbcCost": "NoDocbcCost_bin",
            "GenHlth": "GenHlth_ord",
            "MentHlth": "MentHlthr_numcat",
            "PhysHlth": "PhysHlth_numcat",
            "DiffWalk": "DiffWalk_bin",
            "Sex": "Sex_nom",
            "Age": "Age_nom",
            "Education": "Education_ord",
            "Income": "Income_ord",
        }

        df = df.rename(columns=rename_dict)

        # Sắp xếp các cột theo đúng thứ tự
        (
            numeric_cols,
            numericCat_cols,
            cat_cols,
            binary_cols,
            nominal_cols,
            ordinal_cols,
            target_col,
        ) = myfuncs.get_different_types_cols_from_df_4(df)

        df = df[
            numeric_cols
            + numericCat_cols
            + binary_cols
            + nominal_cols
            + ordinal_cols
            + [target_col]
        ]

        # Chuyển kdl = kdl mong muốn + NAN
        ## Sex_nom
        col_name = "Sex_nom"
        df[col_name] = df[col_name].astype("int").astype("str")
        df[col_name] = myfuncs.replace_in_category_series_33(
            df[col_name],
            [
                (["0"], "female"),
                (["1"], "male"),
            ],
        )

        ## Age_nom
        col_name = "Age_nom"
        df[col_name] = df[col_name].astype("int").astype("str")

        ## GenHlth_ord
        col_name = "GenHlth_ord"
        df[col_name] = df[col_name].astype("int").astype("str")
        df[col_name] = myfuncs.replace_in_category_series_33(
            df[col_name],
            [
                (["1"], "excellent"),
                (["2"], "very_good"),
                (["3"], "good"),
                (["4"], "fair"),
                (["5"], "poor"),
            ],
        )

        ## Education_ord
        col_name = "Education_ord"
        df[col_name] = df[col_name].astype("int").astype("str")
        df[col_name] = myfuncs.replace_in_category_series_33(
            df[col_name],
            [
                (["1"], "never_attend_school_or_only_kindergarten"),
                (["2"], "elementary"),
                (["3"], "high_school"),
                (["4"], "high_school_graduate"),
                (["5"], "college"),
                (["6"], "college_graduate"),
            ],
        )

        ## Income_ord
        col_name = "Income_ord"
        df[col_name] = df[col_name].astype("int").astype("str")

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        self.handler = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numeric_cols),
                (
                    "numCat",
                    SimpleImputer(strategy="most_frequent"),
                    numericCat_cols,
                ),
                ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
                ("target", SimpleImputer(strategy="most_frequent"), [target_col]),
            ]
        )
        self.handler.fit(df)

    def transform(self, X, y=None):
        df = X

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        df = self.handler.transform(df)
        self.cols = numeric_cols + numericCat_cols + cat_cols + [target_col]
        df = pd.DataFrame(df, columns=self.cols)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class AfterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X

        self.cols = df.columns.tolist()

        numeric_cols, numericCat_cols, cat_cols, _, _, _, target_col = (
            myfuncs.get_different_types_cols_from_df_4(df)
        )

        # Chuyển đổi về đúng kiểu dữ liệu
        df[numeric_cols] = df[numeric_cols].astype("float32")
        df[numericCat_cols] = df[numericCat_cols].astype("float32")
        df[cat_cols] = df[cat_cols].astype("category")
        df[target_col] = df[target_col].astype("category")

        # Loại bỏ duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols
