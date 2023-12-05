"""Classification heads with various validation strategies."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2023
ALL RIGHTS RESERVED
"""
import itertools
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

CLASSIFICATION_HEADS = {
    "LDA": LinearDiscriminantAnalysis(),
    "RF": RandomForestClassifier(n_estimators=50, max_depth=None),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": svm.SVC(),
    "ET": ExtraTreesClassifier(),
    "XGB": GradientBoostingClassifier(),
}

min_comp = int(os.getenv("MINIMUM_COMPONENTS_NUMBER", 5))
comp_step = int(os.getenv("COMPONENTS_STEP", 5))
num_rep = int(os.getenv("NUMBER_OF_REPEATS", 50))


def attach_classification_head_kfold(
    features: NDArray[Any],
    class_labels: NDArray[Any],
    model_name: str,
    task_name: str,
    n_folds: int,
    maximum_number_of_components: int,
    output_path: Path,
) -> None:
    """Compute classification results with k-fold validation.

    Args:
        features: array containing features extracted from vision models.
        class_labels: class labels associated to features.
        model_name: name of pre-trained vision model used for feature extraction.
        task_name: name to identify down-stream task.
        n_folds: number of folds used for k-fold cross-validation.
        maximum_number_of_components: maximum number of principal components to be used for model training and validation.
        output_path: output path.
    """
    results_path = Path.joinpath(output_path, "results_classification_kfold", task_name)
    results_path.mkdir(exist_ok=True)
    models = CLASSIFICATION_HEADS.keys()
    classification_heads = deepcopy(CLASSIFICATION_HEADS)

    for model in models:
        Path.joinpath(results_path, model).mkdir(exist_ok=True)

    complete_results = {}
    for n_comp in list(range(min_comp, maximum_number_of_components, comp_step)):
        pca = PCA(n_components=n_comp)
        p_comp = pca.fit_transform(features)
        explanation = np.sum(pca.explained_variance_ratio_)

        x = p_comp
        y = class_labels

        for model in models:
            acc_val = []

            for _ in range(num_rep):
                kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
                kf.get_n_splits(x)

                predictions = []
                validations = []
                best_accuracy = 0

                for train_index, val_index in kf.split(x):
                    x_train, x_val = x[train_index], x[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    classification_heads[model].fit(x_train, y_train)

                    y_predict = classification_heads[model].predict(x_val).tolist()
                    predictions.append(y_predict)
                    validations.append(y_val.tolist())

                    accuracy = accuracy_score(y_val, y_predict)
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy

                y_pred = np.array(list(itertools.chain.from_iterable(predictions)))
                y_true = np.array(list(itertools.chain.from_iterable(validations)))
                pred_df = pd.DataFrame(y_true, columns=["True"])
                pred_df["Predicted"] = y_pred
                pred_df.to_csv(
                    Path.joinpath(
                        results_path,
                        model,
                        f"Predictions_{model}_{task_name}_{str(n_comp)}comp_{str(n_folds)}folds.csv",
                    ),
                    encoding="utf-8",
                    mode="a",
                )

                acc_val.append(accuracy_score(y_true, y_pred))

            overall_acc = np.average(acc_val)
            std_acc = np.std(acc_val)

            results = {
                "NetExtractor": model_name,
                "ModelHead": model,
                "Accuracy": str(overall_acc),
                "StdAccuracy": str(std_acc),
                "PC_components": str(n_comp),
                "PC_explanation": str(explanation),
            }

            complete_results[f"{model_name}_{model}_{str(n_comp)}"] = results

    summary_results = pd.DataFrame.from_dict(complete_results, orient="index")
    summary_results.to_csv(
        Path.joinpath(results_path, f"Accuracy_{task_name}_{str(n_folds)}folds.csv"),
        encoding="utf-8",
        mode="a",
    )


def attach_classification_head_loco(
    features: NDArray[Any],
    class_labels: NDArray[Any],
    model_name: str,
    task_name: str,
    liquid_class: NDArray[Any],
    maximum_number_of_components: int,
    output_path: Path,
) -> None:
    """Compute classification results with leave-one-class-out validation.

    Args:
        features: array containing features extracted from vision models.
        class_labels: class labels associated to features.
        model_name: name of pre-trained vision model used for feature extraction.
        task_name: name to identify down-stream task.
        liquid_class: list of liquid subclasses.
        maximum_number_of_components: maximum number of principal components to be used for model training and validation.
        output_path: output path.
    """
    results_path = Path.joinpath(output_path, "results_classification_loco", task_name)
    results_path.mkdir(exist_ok=True)
    models = CLASSIFICATION_HEADS.keys()
    classification_heads = deepcopy(CLASSIFICATION_HEADS)

    for model in models:
        Path.joinpath(results_path, model).mkdir(exist_ok=True)

    complete_results = {}
    for n_comp in list(range(min_comp, maximum_number_of_components, comp_step)):
        pca = PCA(n_components=n_comp)
        p_comp = pca.fit_transform(features)
        explanation = np.sum(pca.explained_variance_ratio_)

        x = p_comp
        y = class_labels

        for model in models:
            acc_val = []

            for liquid in np.unique(liquid_class):
                predictions = []
                validations = []

                val_index = [
                    i
                    for i, liquid_list in enumerate(liquid_class)
                    if liquid_class[i] == liquid
                ]
                train_index = [
                    ele
                    for ele in list(range(features.shape[0]))
                    if ele not in val_index
                ]

                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]

                classification_heads[model].fit(x_train, y_train)

                y_predict = classification_heads[model].predict(x_val).tolist()
                predictions.append(y_predict)
                validations.append(y_val.tolist())

                accuracy = accuracy_score(y_val, y_predict)

                y_pred = np.array(list(itertools.chain.from_iterable(predictions)))
                y_true = np.array(list(itertools.chain.from_iterable(validations)))
                pred_df = pd.DataFrame(
                    np.array([liquid] * (y_pred.shape)[0]), columns=["Liquid"]
                )
                pred_df["True"] = y_true
                pred_df["Predicted"] = y_pred

                pred_df.to_csv(
                    Path.joinpath(
                        results_path,
                        model,
                        f"Predictions_{model}_{task_name}_{str(n_comp)}_LOCO.csv",
                    ),
                    encoding="utf-8",
                    mode="a",
                )

                acc_val.append(accuracy)

            overall_acc = np.average(acc_val)
            std_acc = np.std(acc_val)

            results = {
                "NetExtractor": model_name,
                "ModelHead": model,
                "Accuracy": str(overall_acc),
                "StdAccuracy": str(std_acc),
                "PC_components": str(n_comp),
                "PC_explanation": str(explanation),
            }

            complete_results[f"{model_name}_{model}_{str(n_comp)}"] = results

    summary_results = pd.DataFrame.from_dict(complete_results, orient="index")
    summary_results.to_csv(
        Path.joinpath(results_path, f"Accuracy_{task_name}_LOCO.csv"),
        encoding="utf-8",
        mode="a",
    )


def attach_classification_head_fewshots(
    features: NDArray[Any],
    class_labels: NDArray[Any],
    model_name: str,
    task_name: str,
    maximum_number_of_components: int,
    output_path: Path,
) -> None:
    """Compute classification results with few-shots training.

    Args:
        features: array containing features extracted from vision models.
        class_labels: class labels associated to features.
        model_name: name of pre-trained vision model used for feature extraction.
        task_name: name to identify down-stream task.
        maximum_number_of_components: maximum number of principal components to be used for model training and validation.
        output_path: output path.
    """
    results_path = Path.joinpath(
        output_path, "results_classification_few_shots", task_name
    )
    results_path.mkdir(exist_ok=True)
    models = CLASSIFICATION_HEADS.keys()
    classification_heads = deepcopy(CLASSIFICATION_HEADS)

    for model in models:
        Path.joinpath(results_path, model).mkdir(exist_ok=True)

    index_class = {}
    for group in np.unique(class_labels):
        index_class[group] = np.array(
            [i for i, s in enumerate(class_labels) if group == s]
        )

    complete_results = {}
    for n_comp in list(range(min_comp, maximum_number_of_components, comp_step)):
        pca = PCA(n_components=n_comp)
        p_comp = pca.fit_transform(features)
        explanation = np.sum(pca.explained_variance_ratio_)

        x = p_comp
        y = class_labels

        for model in models:
            for shot in [3, 5, 7]:
                acc_val = []

                for _ in range(num_rep):
                    t_index = []
                    v_index = []

                    for group in np.unique(class_labels):
                        rand_index = random.sample(list(index_class[group]), shot)
                        t_index.append(rand_index)
                        v_index.append(
                            [
                                ele
                                for ele in list(index_class[group])
                                if ele not in rand_index
                            ]
                        )

                    train_index = np.array(
                        [item for sublist in t_index for item in sublist]
                    )
                    val_index = np.array(
                        [item for sublist in v_index for item in sublist]
                    )

                    predictions = []
                    validations = []
                    best_accuracy = 0

                    x_train, x_val = x[train_index], x[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    classification_heads[model].fit(x_train, y_train)

                    y_predict = classification_heads[model].predict(x_val).tolist()
                    predictions.append(y_predict)
                    validations.append(y_val.tolist())

                    accuracy = accuracy_score(y_val, y_predict)
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy

                    y_pred = np.array(list(itertools.chain.from_iterable(predictions)))
                    y_true = np.array(list(itertools.chain.from_iterable(validations)))
                    pred_df = pd.DataFrame(y_true, columns=["True"])
                    pred_df["Predicted"] = y_pred
                    pred_df.to_csv(
                        Path.joinpath(
                            results_path,
                            model,
                            f"Predictions_{model}_{task_name}_{str(n_comp)}comp_{str(shot)}shots.csv",
                        ),
                        encoding="utf-8",
                        mode="a",
                    )

                    acc_val.append(accuracy_score(y_true, y_pred))

                overall_acc = np.average(acc_val)
                std_acc = np.std(acc_val)

                results = {
                    "NetExtractor": model_name,
                    "ModelHead": model,
                    "Number_Shots": shot,
                    "Accuracy": str(overall_acc),
                    "StdAccuracy": str(std_acc),
                    "PC_components": str(n_comp),
                    "PC_explanation": str(explanation),
                }

                complete_results[
                    f"{model_name}_{model}_{str(n_comp)}_{str(shot)}"
                ] = results

    summary_results = pd.DataFrame.from_dict(complete_results, orient="index")
    summary_results.to_csv(
        Path.joinpath(results_path, f"Accuracy_{task_name}_few_shots.csv"),
        encoding="utf-8",
        mode="a",
    )


def attach_classification_head_loco_sugars(
    features: NDArray[Any],
    class_labels: NDArray[Any],
    model_name: str,
    task_name: str,
    liquid_class: NDArray[Any],
    maximum_number_of_components: int,
    output_path: Path,
) -> None:
    """Compute classification results with leave-one-class-out validation for sugar dataset.

    Args:
        features: array containing features extracted from vision models.
        class_labels: class labels associated to features.
        model_name: name of pre-trained vision model used for feature extraction.
        task_name: name to identify down-stream task.
        liquid_class: list of liquid subclasses.
        maximum_number_of_components: maximum number of principal components to be used for model training and validation.
        output_path: Path
    """
    results_path = Path.joinpath(output_path, "results_classification_loco", task_name)
    results_path.mkdir(exist_ok=True)
    models = CLASSIFICATION_HEADS.keys()
    classification_heads = deepcopy(CLASSIFICATION_HEADS)

    for model in models:
        Path.joinpath(results_path, model).mkdir(exist_ok=True)

    test_liquids = ["2,5_Sucrose", "10_Sucrose", "2,5_Glucose", "10_Glucose"]

    complete_results = {}
    for n_comp in list(range(min_comp, maximum_number_of_components, comp_step)):
        pca = PCA(n_components=n_comp)
        p_comp = pca.fit_transform(features)
        explanation = np.sum(pca.explained_variance_ratio_)

        x = p_comp
        y = class_labels

        for model in models:
            acc_val = []

            for liquid in test_liquids:
                predictions = []
                validations = []

                val_index = [
                    i
                    for i, liquid_list in enumerate(liquid_class)
                    if liquid_class[i] == liquid
                ]
                train_index = [
                    ele
                    for ele in list(range(features.shape[0]))
                    if ele not in val_index
                ]

                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]

                classification_heads[model].fit(x_train, y_train)

                y_predict = classification_heads[model].predict(x_val).tolist()
                predictions.append(y_predict)

                if liquid == "2,5_Sucrose":
                    y_val[y_val == liquid] = "5_Sucrose"
                elif liquid == "10_Sucrose":
                    y_val[y_val == liquid] = "5_Sucrose"
                elif liquid == "2,5_Glucose":
                    y_val[y_val == liquid] = "5_Glucose"
                elif liquid == "10_Glucose":
                    y_val[y_val == liquid] = "5_Glucose"

                validations.append(y_val.tolist())
                accuracy = accuracy_score(y_val, y_predict)

                y_pred = np.array(list(itertools.chain.from_iterable(predictions)))
                y_true = np.array(list(itertools.chain.from_iterable(validations)))
                pred_df = pd.DataFrame(
                    np.array([liquid] * (y_pred.shape)[0]), columns=["Liquid"]
                )
                pred_df["True"] = y_true
                pred_df["Predicted"] = y_pred

                pred_df.to_csv(
                    Path.joinpath(
                        results_path,
                        model,
                        f"Predictions_{model}_{task_name}_{str(n_comp)}comp_LOCO.csv",
                    ),
                    encoding="utf-8",
                    mode="a",
                )

                acc_val.append(accuracy)

            overall_acc = np.average(acc_val)
            std_acc = np.std(acc_val)

            results = {
                "NetExtractor": model_name,
                "ModelHead": model,
                "Accuracy": str(overall_acc),
                "StdAccuracy": str(std_acc),
                "PC_components": str(n_comp),
                "PC_explanation": str(explanation),
            }

            complete_results[f"{model_name}_{model}_{str(n_comp)}"] = results

    summary_results = pd.DataFrame.from_dict(complete_results, orient="index")
    summary_results.to_csv(
        Path(results_path, f"Accuracy_{task_name}_LOCO.csv"),
        encoding="utf-8",
        mode="a",
    )
