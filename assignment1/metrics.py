import pandas as pd
from typing import List
import seaborn as sns
import streamlit as st
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)


class ClassBasedEvaluator:
    def __init__(self, input_image_class: str, retrieved_image_classes: List):
        self.input_image_class = input_image_class
        self.retrieved_image_classes = retrieved_image_classes

    def create_class_matrix(
        self, input_image_class: str, retrieved_image_classes: List
    ) -> pd.DataFrame:
        unique_retrieved_classes = sorted(
            set(retrieved_image_classes + [input_image_class])
        )
        counts = [
            retrieved_image_classes.count(cls) for cls in unique_retrieved_classes
        ]
        class_matrix = pd.DataFrame(
            [counts], index=[input_image_class], columns=unique_retrieved_classes
        )
        class_matrix = class_matrix.sort_values(
            by=input_image_class, axis=1, ascending=False
        )
        return class_matrix

    def plot_class_matrix(self, class_matrix: pd.DataFrame, input_image_class: str):
        # Move the input class to the first position
        columns = list(class_matrix.columns)
        if input_image_class in columns:
            columns.remove(input_image_class)
            columns = [input_image_class] + columns
        class_matrix = class_matrix[columns]

        match_matrix = np.zeros_like(class_matrix, dtype=int)
        for col in class_matrix.columns:
            if col == input_image_class:
                match_matrix[0, class_matrix.columns.get_loc(col)] = 1

        cmap = sns.color_palette(["lightgreen", "pink"])
        plt.figure(figsize=(10, 2))
        if class_matrix[input_image_class].iloc[0] == 0:
            sns.heatmap(
                class_matrix,
                annot=True,
                fmt="d",
                cmap=["pink"],
                cbar=False,
                linewidths=0.5,
                linecolor="black",
            )
        else:
            sns.heatmap(
                class_matrix,
                annot=True,
                fmt="d",
                cmap=cmap,
                cbar=False,
                linewidths=0.5,
                linecolor="black",
                mask=(match_matrix == 0),
            )
            sns.heatmap(
                class_matrix,
                annot=True,
                fmt="d",
                cmap=["pink"],
                cbar=False,
                linewidths=0.5,
                linecolor="black",
                mask=(match_matrix == 1),
            )
        plt.xlabel("Retrieved Classes")
        plt.ylabel("Input Class")
        plt.tight_layout()
        st.pyplot(plt)

    def calculate_precision_recall(
        self,
        input_image_class: str,
        retrieved_image_classes: List,
        total_relevant_images: int,
    ):
        tp = retrieved_image_classes.count(input_image_class)
        fp = len(retrieved_image_classes) - tp
        fn = total_relevant_images - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (precision, recall)

    def calculate_pr_curve(
        self,
        input_image_class: str,
        retrieved_image_classes: list,
        total_relevant_images: int,
    ):
        precision_values = []
        recall_values = []
        for i in range(1, len(retrieved_image_classes) + 1):
            top_retrieved_classes = retrieved_image_classes[:i]
            precision, recall = self.calculate_precision_recall(
                input_image_class, top_retrieved_classes, total_relevant_images
            )
            precision_values.append(precision)
            recall_values.append(recall)
        return precision_values, recall_values

    def plot_pr_curve(self, precision_values: list, recall_values: list):
        thresholds = list(range(1, len(precision_values) + 1))  # Top N values
        data = pd.DataFrame(
            {
                "Precision": precision_values,
                "Recall": recall_values,
                "Threshold": thresholds,
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["Recall"],
                y=data["Precision"],
                mode="lines+markers",
                text=data["Threshold"],
                hovertemplate="Precision: %{y}<br>Recall: %{x}<br>Threshold: %{text}<extra></extra>",
                name="PR Curve",
            )
        )
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            hovermode="closest",
        )
        st.plotly_chart(fig)


class LabelBasedEvaluator:
    def __init__(
        self, input_image_labels: List[str], retrieved_image_labels: List[List[str]]
    ):
        self.input_image_labels = input_image_labels
        self.retrieved_image_labels = retrieved_image_labels

    def create_labels_matrix(
        self, input_image_labels: List[str], retrieved_image_labels: List[List[str]]
    ) -> pd.DataFrame:
        input_image_labels = sorted(input_image_labels)
        unique_retrieved_labels = sorted(
            set(input_image_labels).union(*retrieved_image_labels)
        )
        labels_matrix = []
        input_labels_set = set(input_image_labels)
        for label in unique_retrieved_labels:
            row = []
            for retrieved_labels in retrieved_image_labels:
                retrieved_labels_set = set(retrieved_labels)
                if label in input_labels_set and label in retrieved_labels_set:
                    row.append(1)
                elif label not in input_labels_set and label in retrieved_labels_set:
                    row.append(-1)
                else:
                    row.append(0)
            labels_matrix.append(row)
        labels_df = pd.DataFrame(
            labels_matrix,
            index=unique_retrieved_labels,
            columns=[f"Image {i+1}" for i in range(len(retrieved_image_labels))],
        )
        return labels_df

    def plot_labels_matrix(self, labels_df: pd.DataFrame):
        plt.figure(figsize=(10, 4))
        cmap = sns.color_palette(["pink", "white", "lightgreen"], as_cmap=True)
        sns.heatmap(
            labels_df,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="black",
        )
        plt.xlabel("Retrieved Images")
        plt.ylabel("Labels")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(plt)
