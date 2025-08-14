from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Put model in evaluation mode
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:  # test_loader should load your test dataset
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # forward pass
        _, predicted = torch.max(outputs, 1)  # get class index with highest score
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy
print("Accuracy:", accuracy_score(y_true, y_pred))

# F1 Score (macro = all classes equally)
print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))

# F1 Score (weighted = considers class imbalance)
print("F1 Score (weighted):", f1_score(y_true, y_pred, average='weighted'))

# Classification report (precision, recall, f1 for each class)
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
