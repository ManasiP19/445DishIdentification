import pandas as pd
from pathlib import Path

def generate_confusion_matrix(precision, recall, f1_score, support):
    tp = precision * support
    fn = support - tp
    fp = tp / precision - tp
    tn = total_instances - (tp + fp + fn)

    confusion_matrix = [[int(tn), int(fp)],
                        [int(fn), int(tp)]]
    
    return confusion_matrix

# Specify the file paths using pathlib.Path
file_path_model1 = Path('tensorflow_info.csv')
file_path_model2 = Path('googlenet_info.csv')

# Read CSV files for each model
df_model1 = pd.read_csv(file_path_model1)
df_model2 = pd.read_csv(file_path_model2)

# Extract precision, recall, f1-score, and support for each model
precision_model1 = df_model1['precision'].values[0]
recall_model1 = df_model1['recall'].values[0]
f1_score_model1 = df_model1['f1score'].values[0]
support_model1 = df_model1['support'].values[0]

precision_model2 = df_model2['precision'].values[0]
recall_model2 = df_model2['recall'].values[0]
f1_score_model2 = df_model2['f1score'].values[0]
support_model2 = df_model2['support'].values[0]

# Calculate total instances for both models
total_instances = support_model1 + support_model2

# Generate confusion matrix for each model
confusion_matrix_model1 = generate_confusion_matrix(precision_model1, recall_model1, f1_score_model1, support_model1)
confusion_matrix_model2 = generate_confusion_matrix(precision_model2, recall_model2, f1_score_model2, support_model2)

# Display confusion matrices
print("Confusion Matrix for Model 1:")
print(confusion_matrix_model1)

print("\nConfusion Matrix for Model 2:")
print(confusion_matrix_model2)
