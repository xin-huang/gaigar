# Copyright 2024 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from gaia.utils.models import MLModel

pd.options.mode.chained_assignment = None


class LRModel(MLModel):
    """
    A logistic regression model class for training and inferring purposes,
    specifically designed for datasets in genomics or biological contexts.
    This class provides static methods to train a logistic regression model
    and to perform inference with an existing model.

    """

    @staticmethod
    def train(
        training_data: str,
        model_file: str,
        seed: int = None,
        solver: str = "newton-cg",
        penalty: str = None,
        max_iter: int = 10000,
        is_scaled: bool = False,
    ) -> None:
        """
        Train a logistic regression model using provided training data,
        save the model and the scaler (if data scaling is applied) to disk.
        If `is_scaled` is True, the feature data will be scaled using StandardScaler
        and a scaler object will be saved to disk alongside the model, with the filename
        `<model_file>.scaler`.

        Parameters
        ----------
        training_data : str
            Path to the training data file in tab-separated format.
        model_file : str
            Path where the trained model will be saved.
        seed : int, optional
            Random seed for reproducibility. Default: None.
        solver : str, optional
            Solver to use in the logistic regression algorithm. Default: "newton-cg".
        penalty : str, optional
            Regularization term to use. Default: None.
        max_iter : int, optional
            Maximum number of iterations for the solver to converge. Default: 10000.
        is_scaled : bool, optional
            Indicates whether the feature data should be scaled. Default: False.

        """
        features = pd.read_csv(training_data, sep="\t")
        output_dir = os.path.dirname(model_file)
        os.makedirs(output_dir, exist_ok=True)

        labels = features["Label"]
        data = features.drop(
            columns=["Chromosome", "Start", "End", "Sample", "Replicate", "Label"]
        ).values

        if is_scaled:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            joblib.dump(scaler, f"{model_file}.scaler")

        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            max_iter=max_iter,
            random_state=seed,
        )
        model.fit(data, labels.astype(int))

        joblib.dump(model, model_file)

    @staticmethod
    def infer(
        inference_data: str, model_file: str, output_file: str, is_scaled: bool = False
    ) -> None:
        """
        Perform inference using a trained logistic regression model on new data, outputting
        predictions to a specified file. If `is_scaled` is True, it loads the scaler object
        from `<model_file>.scaler` and applies it to scale the feature data before inference.

        Parameters
        ----------
        inference_data : str
            Path to the inference data file in tab-separated format.
        model_file : str
            Path to the saved trained model. The method will also look for `<model_file>.scaler`
            if `is_scaled` is True to load and apply the scaler.
        output_file : str
            Path where the inference output will be saved.
        is_scaled : bool, optional
            If True, scales the feature data using the scaler object saved during training,
            which is expected to be found at `<model_file>.scaler`. Default: False.

        """
        features = pd.read_csv(inference_data, sep="\t")
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        data = features.drop(columns=["Chromosome", "Start", "End", "Sample"]).values

        if is_scaled:
            scaler = joblib.load(f"{model_file}.scaler")
            data = scaler.transform(data)

        model = joblib.load(model_file)

        predictions = model.predict_proba(data)
        prediction_df = features[["Chromosome", "Start", "End", "Sample"]]

        class_names = {
            "0": "Non_Intro",
            "1": "Intro",
        }

        classes = model.classes_
        for i in range(len(classes)):
            class_name = class_names[f"{classes[i]}"]
            prediction_df[f"{class_name}_Prob"] = predictions[:, i]

        prediction_df.sort_values(by=["Sample", "Chromosome", "Start", "End"]).to_csv(
            output_file, sep="\t", index=False
        )
