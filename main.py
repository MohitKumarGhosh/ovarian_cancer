import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# Load data
data = pd.read_csv("csv_result-Ovarian_fixed.csv")

# Ensure 'Class' is numeric
if data["Class"].dtype == 'object':
    data["Class"] = data["Class"].map({"Normal": 0, "Cancer": 1})  # Map "Normal" to 0 and "Cancer" to 1

# Separate features and target
X = data.drop(columns=["Class"])
y = data["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the neural network
class OvarianCancerPredictor(nn.Module):
    def _init_(self, input_size):
        super(OvarianCancerPredictor, self)._init_()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model
input_size = X_train.shape[1]
model = OvarianCancerPredictor(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log loss for every epoch
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation and confusion matrix
with torch.no_grad():
    model.eval()
    predictions = model(X_test_tensor).squeeze()
    predictions = (predictions > 0.5).float()  # Apply threshold of 0.5

# Convert predictions and true labels to numpy arrays
y_pred = predictions.numpy()
y_true = y_test_tensor.numpy()

# Handle potential single-class issue by specifying all known labels
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Cancer"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Print accuracy
accuracy = (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
from scipy.stats import ttest_ind

feature_t_tests = {}
for feature in X.columns:
    normal_values = data[data["Class"] == 0][feature]
    cancer_values = data[data["Class"] == 1][feature]
    t_stat, p_value = ttest_ind(normal_values, cancer_values, equal_var=False)  # Welch's t-test
    feature_t_tests[feature] = (t_stat, p_value)

# Display features with significant differences (p-value < 0.05)
print("\nFeatures with significant differences between Normal and Cancer:")
for feature, (t_stat, p_value) in feature_t_tests.items():
    if p_value < 0.05:
        print(f"{feature}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
# Calculate correlation of all features with the target 'Class'
correlations = data.corrwith(data["Class"])

# Select top 50 features with the highest absolute correlation
top_features = correlations.abs().nlargest(50).index
top_corr_matrix = data[top_features].corr()

# Plot heatmap for top 50 features
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(top_corr_matrix, annot=False, cmap="coolwarm", fmt='.2f')
plt.title("Pearson Correlation Heatmap (Top 50 Features)")
plt.show()
from scipy.stats import ttest_ind

# Perform t-test for each feature
t_test_results = {}
for feature in X.columns:
    normal_values = data[data["Class"] == 0][feature]
    cancer_values = data[data["Class"] == 1][feature]
    t_stat, p_value = ttest_ind(normal_values, cancer_values, equal_var=False)  # Welch's t-test
    t_test_results[feature] = (t_stat, p_value)

# Filter significant features (p-value < 0.05)
significant_t_features = {feature: stats for feature, stats in t_test_results.items() if stats[1] < 0.05}

print(f"\nNumber of significant features (T-test): {len(significant_t_features)}")
for feature, (t_stat, p_value) in significant_t_features.items():
    print(f"{feature}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
from scipy.stats import f_oneway

# Perform ANOVA for each feature
anova_results = {}
for feature in X.columns:
    normal_values = data[data["Class"] == 0][feature]
    cancer_values = data[data["Class"] == 1][feature]
    f_stat, p_value = f_oneway(normal_values, cancer_values)
    anova_results[feature] = (f_stat, p_value)

# Filter significant features (p-value < 0.05)
significant_anova_features = {feature: stats for feature, stats in anova_results.items() if stats[1] < 0.05}

print(f"\nNumber of significant features (ANOVA): {len(significant_anova_features)}")
for feature, (f_stat, p_value) in significant_anova_features.items():
    print(f"{feature}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4e}")
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency

# Discretize continuous features into bins
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_binned = pd.DataFrame(binner.fit_transform(X), columns=X.columns)

# Perform Chi-Square test
chi_square_results = {}
for feature in X_binned.columns:
    contingency_table = pd.crosstab(X_binned[feature], data["Class"])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    chi_square_results[feature] = (chi2_stat, p_value)

# Filter significant features (p-value < 0.05)
significant_chi_features = {feature: stats for feature, stats in chi_square_results.items() if stats[1] < 0.05}

print(f"\nNumber of significant features (Chi-Square): {len(significant_chi_features)}")
for feature, (chi2_stat, p_value) in significant_chi_features.items():
    print(f"{feature}: Chi2-statistic = {chi2_stat:.4f}, p-value = {p_value:.4e}")
