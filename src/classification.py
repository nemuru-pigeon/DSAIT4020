from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV, validation_curve
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
    # Check if saved models exist
    if os.path.exists(RESULTS_FILE) and os.path.exists(PREDS_FILE):
        print("Loading saved models...")
        results = joblib.load(RESULTS_FILE)
        model_preds = joblib.load(PREDS_FILE)
        learning_curves = joblib.load(os.path.join(MODEL_DIR, "learning_curves.joblib"))
        validation_curves = joblib.load(os.path.join(MODEL_DIR, "validation_curves.joblib"))
        return results, model_preds, learning_curves, validation_curves
    else:
        print("Training models...")
        
        results = []
        model_preds = {}
        learning_curves = {} 
        validation_curves = {}  # Dictionary to store validation curve data
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

            # Compute learning curve
            best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model.named_steps['model'])])
            train_sizes, train_scores, test_scores = learning_curve(
                best_pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy',
                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
            )

            learning_curves[model_name] = {
                'train_sizes': train_sizes,
                'train_scores': train_scores,
                'test_scores': test_scores
            }

            # Compute validation curve with model-specific parameters
            if model_name == 'Logistic Regression' or model_name == 'Support Vector Machine':
                param_name = 'model__C'
                param_range = np.logspace(-3, 3, 7)
            elif model_name == 'Random Forest':
                param_name = 'model__n_estimators'
                param_range = np.array([10, 30, 50, 100, 200, 300])
            elif model_name == 'Decision Tree':
                param_name = 'model__max_depth'
                param_range = np.array([1, 3, 5, 7, 9, 11, 13])
            
            train_scores_val, test_scores_val = validation_curve(
                best_pipeline, X_train, y_train, 
                param_name=param_name, 
                param_range=param_range,
                cv=cv_folds, 
                scoring='accuracy', 
                n_jobs=-1
            )

            # Store validation curve data
            validation_curves[model_name] = {
                'param_range': param_range,
                'param_name': param_name,  # Store parameter name for plotting
                'train_scores': train_scores_val,
                'test_scores': test_scores_val
            }

            # Plot and save the learning curve
            plt.figure()
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score")
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation Score")
            plt.xlabel("Training Size")
            plt.ylabel("Accuracy")
            plt.title(f"Learning Curve - {model_name}")
            plt.legend()
            plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_learning_curve.png"))
            plt.close()

            # Plot and save the validation curve
            plt.figure()
            plt.semilogx(param_range, np.mean(train_scores_val, axis=1), label="Training Score")
            plt.semilogx(param_range, np.mean(test_scores_val, axis=1), label="Validation Score")
            plt.xlabel(f"Parameter {param_name.split('__')[1]}")  # Remove 'model__' prefix
            plt.ylabel("Accuracy")
            plt.title(f"Validation Curve - {model_name}")
            plt.legend()
            plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_validation_curve.png"))
            plt.close()

            # Save trained model
            model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
            joblib.dump(best_model, model_path)

        # Save results
        joblib.dump(results, RESULTS_FILE)
        joblib.dump(model_preds, PREDS_FILE)
        joblib.dump(learning_curves, os.path.join(MODEL_DIR, "learning_curves.joblib"))
        joblib.dump(validation_curves, os.path.join(MODEL_DIR, "validation_curves.joblib"))
        print("Models and learning curves saved successfully.")

        return results, model_preds, learning_curves, validation_curves

if __name__ == "__main__":
    setup_environment()

    df_data = load_and_preprocess_data('speeddating.arff')
    X_train, X_test, y_train, y_test = get_train_test_split(df_data)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    results, model_preds, learning_curves, validation_curves = load_or_train_models(X_train, X_test, y_train, y_test)

    # Dashboard
    print("Initializing dashboard...")
    dash_app = Dashboard(results, learning_curves, validation_curves)
    
    print("Defining layout...")
    dash_app.define_layout(y_test, model_preds)
    
    print("After initializing dashboard")
    dash_app.run()
    print("Closing programm")
