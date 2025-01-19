import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from classification_experiment_settings import models, param_grids
from dataloader import load_arff_to_dataframe

# Load dataset
# Replace this with the actual loading process for your dataset
# df = pd.read_csv('../data/speeddating.csv')
df = load_arff_to_dataframe('../speeddating.arff')

print(df.head())

# Features and target
X = df[['d_age', 'samerace', 'pref_o_attractive', 'pref_o_funny', 'pref_o_ambitious', 'interests_correlate']]
y = df['match']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['d_age', 'pref_o_attractive', 'pref_o_funny', 'pref_o_ambitious', 'interests_correlate']
categorical_features = ['samerace']

# Standardizing numeric features and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Evaluate models
results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Append results
    results.append({
        'Model': model_name,
        'Best Params': grid_search.best_params_,
        'Accuracy': acc,
        'Classification Report': report
    })

# Display results
for result in results:
    print(f"Model: {result['Model']}")
    print(f"Best Params: {result['Best Params']}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Classification Report: {result['Classification Report']}")
    print("-" * 50)
