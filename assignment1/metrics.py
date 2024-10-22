import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st

def create_class_matrix(input_image_class: str, retrieved_image_classes: list) -> pd.DataFrame:
    unique_retrieved_classes = sorted(set(retrieved_image_classes + [input_image_class]))
    counts = [retrieved_image_classes.count(cls) for cls in unique_retrieved_classes]
    confusion_df = pd.DataFrame([counts], index=[input_image_class], columns=unique_retrieved_classes)
    return confusion_df

def plot_class_matrix(confusion_df: pd.DataFrame):
    plt.figure(figsize=(10, 2))
    sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel("Retrieved Classes")
    plt.ylabel("Input Class")
    plt.tight_layout()
    st.pyplot(plt)
