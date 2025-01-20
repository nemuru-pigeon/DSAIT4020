import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from classification_experiment_settings import models, param_grids, required_columns, numeric_features, \
    categorical_features
from dataloader import load_arff_to_dataframe
from src.util import fill_nan_values
from dashboard import Dashboard

import os

# File paths for saving models and results
MODEL_DIR = "saved_models"
RESULTS_FILE = os.path.join(MODEL_DIR, "model_results.pkl")
PREDS_FILE = os.path.join(MODEL_DIR, "model_preds.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


# Print all rows and columns when printing a pandas Dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load dataset
df = load_arff_to_dataframe('../speeddating.arff')

# Exclude derived "delta" columns of which the value can be derived from the values of another column
df = df[required_columns]

# Fill the empty values of all remaining columns. For more info, see the fill_nan_values method
df = fill_nan_values(df)

# Features and target columns
y = df['match'].astype(int)
X = df.drop(columns=['like', 'guess_prob_liked', 'met', 'decision', 'match'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# For all models:
# setup the pipeline, apply grid search
# Store best_model, predictions on test data, y_scores, accuracy on test data, and the classification_report
# Check if saved models exist
if os.path.exists(RESULTS_FILE) and os.path.exists(PREDS_FILE):
    print("Loading saved models...")
    results = joblib.load(RESULTS_FILE)
    model_preds = joblib.load(PREDS_FILE)
else:
    print("Training models...")
    # Evaluate models
    results = []
    model_preds = {}
    cv_folds = 5

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Save trained model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(best_model, model_path)

        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            'Model': model_name,
            'Best Params': grid_search.best_params_,
            'Accuracy': acc,
            'Classification Report': report
        })

        model_preds[model_name] = {'y_pred': y_pred, 'y_scores': y_scores}

    # Save results
    joblib.dump(results, RESULTS_FILE)
    joblib.dump(model_preds, PREDS_FILE)
    print("Models saved successfully.")

# Launch dashboard
print("Before initializing dashboard")
dash_app = Dashboard(results)
print("After initializing dashboard")
dash_app.define_layout(y_test, model_preds)
dash_app.run()
print("Finish running dashboard")
