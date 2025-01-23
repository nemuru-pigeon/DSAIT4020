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
from util import fill_nan_values
from dashboard import Dashboard

import os


# File paths for saving models and results
MODEL_DIR = "saved_models"
RESULTS_FILE = os.path.join(MODEL_DIR, "model_results.pkl")
PREDS_FILE = os.path.join(MODEL_DIR, "model_preds.pkl")

def setup_environment():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Print all rows and columns when printing a pandas Dataframe
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

def load_and_preprocess_data(path_to_data) -> pd.DataFrame:
    # Load dataset
    df = load_arff_to_dataframe(path_to_data)
    # Exclude derived "delta" columns of which the value can be derived from the values of another column
    df = df[required_columns]
    # Fill the empty values of all remaining columns. For more info, see the fill_nan_values method
    df = fill_nan_values(df)
    return df

def get_train_test_split(df: pd.DataFrame):
    # Features and target columns
    y = df['match'].astype(int)
    X = df.drop(columns=['like', 'guess_prob_liked', 'met', 'decision', 'match'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_or_train_models(X_train, X_test, y_train, y_test):
    # For all models:
    # setup the pipeline, apply grid search
    # Store best_model, predictions on test data, y_scores, accuracy on test data, and the classification_report
    # Check if saved models exist
    if os.path.exists(RESULTS_FILE) and os.path.exists(PREDS_FILE):
        print("Loading saved models...")
        results = joblib.load(RESULTS_FILE)
        model_preds = joblib.load(PREDS_FILE)
        return results, model_preds
    else:
        print("Training models...")
        # Evaluate models
        results = []
        model_preds = {}
        cv_folds = 5

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Apply grid search to model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=cv_folds, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get grid search results
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            y_pred = best_model.predict(X_test)
            y_scores = best_model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            results.append({
                'Model': model_name,
                'Best Params': best_params,
                'Accuracy': acc,
                'Classification Report': report
            })

            model_preds[model_name] = {'y_pred': y_pred, 'y_scores': y_scores}

            # Save trained model
            model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
            joblib.dump(best_model, model_path)

        # Save results
        joblib.dump(results, RESULTS_FILE)
        joblib.dump(model_preds, PREDS_FILE)
        print("Models saved successfully.")

        return results, model_preds

if __name__ == "__main__":
    setup_environment()

    df_data = load_and_preprocess_data('speeddating.arff')
    X_train, X_test, y_train, y_test = get_train_test_split(df_data)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    results, model_preds = load_or_train_models(X_train, X_test, y_train, y_test)

    # Dashboard
    print("Initializing dashboard...")
    dash_app = Dashboard(results)
    
    print("Defining layout...")
    dash_app.define_layout(y_test, model_preds)
    
    print("After initializing dashboard")
    dash_app.run()
    print("Closing programm")
