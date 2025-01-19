import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from classification_experiment_settings import models, param_grids
from dataloader import load_arff_to_dataframe
from src.util import fill_nan_values

# Load dataset
# Replace this with the actual loading process for your dataset
# df = pd.read_csv('../data/speeddating.csv')
df = load_arff_to_dataframe('../speeddating.arff')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# self_rated_features = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition']
#
# Removed column values can be inferred from kept columns
required_columns = [
    'wave', 'gender', 'age', 'age_o', 'race', 'race_o', 'importance_same_race',
    'importance_same_religion', 'field', 'pref_o_attractive', 'pref_o_sincere',
    'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious',
    'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o',
    'funny_o', 'ambitous_o', 'shared_interests_o', 'attractive_important',
    'sincere_important', 'intellicence_important', 'funny_important',
    'ambtition_important', 'shared_interests_important', 'attractive',
    'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner',
    'sincere_partner', 'intelligence_partner', 'funny_partner',
    'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports',
    'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
    'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping',
    'yoga', 'interests_correlate', 'expected_happy_with_sd_people',
    'expected_num_interested_in_me', 'expected_num_matches', 'like',
    'guess_prob_liked', 'met', 'decision', 'match'
]
df = df[required_columns]
df = fill_nan_values(df)

# Features and target
y = df['match']
X = df.drop(columns=['like', 'guess_prob_liked', 'met', 'decision', 'match'])
# X = df[['d_age', 'samerace', 'pref_o_attractive', 'pref_o_funny', 'pref_o_ambitious', 'interests_correlate']]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
# numeric_features = ['d_age', 'pref_o_attractive', 'pref_o_funny', 'pref_o_ambitious', 'interests_correlate']
# categorical_features = ['samerace']
categorical_features = ['gender', 'race', 'race_o', 'field']
numeric_features = ['wave', 'age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive',
                    'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious',
                    'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o',
                    'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important',
                    'funny_important', 'ambtition_important', 'shared_interests_important', 'attractive', 'sincere',
                    'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner'
                    , 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise',
                    'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
                    'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate',
                    'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches']

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
