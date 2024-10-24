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

    def count_total_relevant_images(self, selected_image, labels_dict) -> int:
        count = 0
        for k, v in labels_dict.items():
            img_class = v["class"]
            # exclude the selected image
            if k == selected_image:
                continue
            if img_class == self.input_image_class:
                count += 1
        return count

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
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig)


class LabelBasedEvaluator:
    def __init__(
        self, input_image_labels: List[str], retrieved_image_labels: List[List[str]]
    ):
        self.input_image_labels = input_image_labels
        self.retrieved_image_labels = retrieved_image_labels

    def create_labels_matrix(self) -> pd.DataFrame:
        input_image_labels = sorted(self.input_image_labels)
        unique_retrieved_labels = sorted(
            set(input_image_labels).union(*self.retrieved_image_labels)
        )
        labels_matrix = []
        input_labels_set = set(input_image_labels)
        for label in unique_retrieved_labels:
            row = []
            for retrieved_labels in self.retrieved_image_labels:
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
            columns=[f"Image {i+1}" for i in range(len(self.retrieved_image_labels))],
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
            vmin=-1,
            vmax=1,
        )
        plt.xlabel("Retrieved Images")
        plt.ylabel("Labels")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    def count_total_relevant_images(self, selected_image, labels_dict) -> int:
        count = 0
        for k, v in labels_dict.items():
            img_labels = v["labels"]
            # exclude the selected image
            if k == selected_image:
                continue
            if any(label in self.input_image_labels for label in img_labels):
                count += 1
        return count

    def calculate_cumulative_precision_recall(
        self, total_relevant_images: int
    ) -> pd.DataFrame:
        labels_matrix = self.create_labels_matrix()
        precision_list = []
        recall_list = []
        cumulative_correct_images = 0
        cumulative_retrieved_images = 0

        for i in range(labels_matrix.shape[1]):
            column = labels_matrix.iloc[:, i]
            if (column == 1).any():
                cumulative_correct_images += 1
            cumulative_retrieved_images += 1
            # Precision = relevant retrieved images / total retrieved images
            precision = (
                cumulative_correct_images / cumulative_retrieved_images
                if cumulative_retrieved_images > 0
                else 0
            )
            precision_list.append(precision)
            # Recall = relevant retrieved images / total relevant images in the dataset
            recall = (
                cumulative_correct_images / total_relevant_images
                if total_relevant_images > 0
                else 0
            )
            recall_list.append(recall)
        pr_df = pd.DataFrame(
            {
                "Image Index": [
                    i + 1 for i in range(labels_matrix.shape[1])
                ],  # Index of retrieved images
                "Cumulative Precision": precision_list,
                "Cumulative Recall": recall_list,
            }
        )
        return pr_df

    def plot_pr_curve(self, total_relevant_images):
        pr_df = self.calculate_cumulative_precision_recall(total_relevant_images)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pr_df["Cumulative Recall"],
                y=pr_df["Cumulative Precision"],
                mode="lines+markers",
                name="Precision-Recall Curve",
                hovertemplate="Precision: %{y}<br>Recall: %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Precision-Recall Curve (Instance-Level Evaluation)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            hovermode="closest",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig)
