def fill_nan_values(df_original):
    df = df_original.copy()
    import pandas as pd
    import numpy as np

    # Assuming df is the cleaned DataFrame from the previous step
    # If not already available, you can recreate it as described earlier.

    # Handling missing values
    # Imputation strategy for each type of column
    for col in df.columns:
        if df[col].dtype == 'object':
            # For categorical columns, fill NaN with the mode (most frequent value)
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif col in ['age', 'age_o']:
            # For age-related columns, fill NaN with the median, as age is typically not skewed
            df[col].fillna(df[col].median(), inplace=True)
        elif col in ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
                     'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']:
            # For preferences, use mean as these are likely on a numerical scale
            df[col].fillna(df[col].mean(), inplace=True)
        elif col in ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
                     'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts',
                     'music', 'shopping', 'yoga']:
            # For activity interest scores, use mean to retain the overall distribution
            df[col].fillna(df[col].mean(), inplace=True)
        elif col in ['like', 'guess_prob_liked', 'expected_happy_with_sd_people',
                     'expected_num_interested_in_me', 'expected_num_matches']:
            # For subjective scores or expectations, use the mean
            df[col].fillna(df[col].mean(), inplace=True)
        elif col in ['decision', 'match', 'met']:
            # For binary columns, fill with 0, assuming no interaction/match is the default
            df[col].fillna(0, inplace=True)
        else:
            # For any other numeric columns, use mean as a general strategy
            df[col].fillna(df[col].mean(), inplace=True)

    # Display the DataFrame to ensure no missing values remain
    print(df.isnull().sum())
    return df
