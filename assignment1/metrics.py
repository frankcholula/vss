import pandas as pd
from typing import List
import seaborn as sns
import streamlit as st
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def create_class_matrix(input_image_class: str, retrieved_image_classes: List) -> pd.DataFrame:
    unique_retrieved_classes = sorted(set(retrieved_image_classes + [input_image_class]))
    counts = [retrieved_image_classes.count(cls) for cls in unique_retrieved_classes]
    confusion_df = pd.DataFrame([counts], index=[input_image_class], columns=unique_retrieved_classes)
    confusion_df = confusion_df.sort_values(by=input_image_class, axis=1, ascending=False)
    return confusion_df

def plot_class_matrix(confusion_df: pd.DataFrame, input_image_class: str):
    # Move the input class to the first position
    columns = list(confusion_df.columns)
    if input_image_class in columns:
        columns.remove(input_image_class)
        columns = [input_image_class] + columns
    confusion_df = confusion_df[columns]

    match_matrix = np.zeros_like(confusion_df, dtype=int)
    for col in confusion_df.columns:
        if col == input_image_class:
            match_matrix[0, confusion_df.columns.get_loc(col)] = 1
    
    cmap = sns.color_palette(['lightgreen', 'pink'])
    plt.figure(figsize=(10, 2))
    if confusion_df[input_image_class].iloc[0] == 0:
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap=['pink'], cbar=False, linewidths=0.5, linecolor='black')
    else:
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=0.5, linecolor='black', mask=(match_matrix == 0))
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap=['pink'], cbar=False, linewidths=0.5, linecolor='black', mask=(match_matrix == 1))
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

def calculate_pr_curve(input_image_class: str, retrieved_image_classes: list, total_relevant_images:int):
    precision_values = []
    recall_values = []
    for i in range(1, len(retrieved_image_classes) + 1):
        top_retrieved_classes = retrieved_image_classes[:i]        
        precision, recall = calculate_precision_recall(input_image_class, top_retrieved_classes, total_relevant_images)
        precision_values.append(precision)
        recall_values.append(recall)
    return precision_values, recall_values

def plot_pr_curve(precision_values: list, recall_values: list):
    thresholds = list(range(1, len(precision_values) + 1))  # Top N values
    data = pd.DataFrame({
        'Precision': precision_values,
        'Recall': recall_values,
        'Threshold': thresholds
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Recall'],
        y=data['Precision'],
        mode='lines+markers', 
        text=data['Threshold'],
        hovertemplate='Precision: %{y}<br>Recall: %{x}<br>Threshold: %{text}<extra></extra>',
        name='PR Curve'
    ))

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        hovermode='closest'
    )
    st.plotly_chart(fig)
