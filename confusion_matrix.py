import pandas as pd

# Read CSV files for each model
df_model1 = pd.read_csv('tensorflow_info.csv')
df_model2 = pd.read_csv('googlenet_info.csv')

# Extract precision, recall, f1-score, and support for each model
name1 = df_model1['name'].values[0]
precision_model1 = df_model1['precision'].values[0]
recall_model1 = df_model1['recall'].values[0]
f1_score_model1 = df_model1['f1-score'].values[0]
support_model1 = df_model1['support'].values[0]

name2 = df_model2['name'].values[0]
precision_model2 = df_model2['precision'].values[0]
recall_model2 = df_model2['recall'].values[0]
f1_score_model2 = df_model2['f1-score'].values[0]
support_model2 = df_model2['support'].values[0]

total_instances = support_model1 + support_model2

# Function to generate confusion matrix
def generate_confusion_matrix(precision, recall, f1_score, support):
    tp = precision * support
    fn = support - tp
    fp = tp / precision - tp
    tn = total_instances - (tp + fp + fn)

    confusion_matrix = [[int(tn), int(fp)],
                        [int(fn), int(tp)]]
    
    return confusion_matrix

# Generate confusion matrix for each model
confusion_matrix_model1 = generate_confusion_matrix(precision_model1, recall_model1, f1_score_model1, support_model1)
confusion_matrix_model2 = generate_confusion_matrix(precision_model2, recall_model2, f1_score_model2, support_model2)

print("Confusion Matrix for Model 1:")
print(confusion_matrix_model1)

print("\nConfusion Matrix for Model 2:")
print(confusion_matrix_model2)
