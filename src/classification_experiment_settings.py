from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Model definitions
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Hyperparameter grids
param_grids = {
    'Logistic Regression': {'model__C': [0.1, 1.0, 10]},
    'Random Forest': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]},
    'Support Vector Machine': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
    'Decision Tree': {'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5, 10]}
}