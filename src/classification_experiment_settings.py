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

# Preprocessing pipeline
categorical_features = ['gender', 'race', 'race_o', 'field']
numeric_features = [
    'wave', 'age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive',
    'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious',
    'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o',
    'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important',
    'funny_important', 'ambtition_important', 'shared_interests_important', 'attractive', 'sincere',
    'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner',
    'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise',
    'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
    'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate',
    'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches'
]