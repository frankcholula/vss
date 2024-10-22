import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
from typing import List

def create_class_matrix(input_image_class: str, retrieved_image_classes: List) -> pd.DataFrame:
    unique_retrieved_classes = sorted(set(retrieved_image_classes + [input_image_class]))
    counts = [retrieved_image_classes.count(cls) for cls in unique_retrieved_classes]
    confusion_df = pd.DataFrame([counts], index=[input_image_class], columns=unique_retrieved_classes)
    confusion_df = confusion_df.sort_values(by=input_image_class, axis=1, ascending=False)
    return confusion_df

def plot_class_matrix(confusion_df: pd.DataFrame):
    plt.figure(figsize=(8, 2))
    sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel("Retrieved Classes")
    plt.ylabel("Input Class")
    plt.tight_layout()
    st.pyplot(plt)

def calculate_precision_recall(input_image_class: str, retrieved_image_classes: List, total_relevant_images:int):
    tp = retrieved_image_classes.count(input_image_class)
    fp = len(retrieved_image_classes) - tp
    fn = total_relevant_images - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (precision, recall)
