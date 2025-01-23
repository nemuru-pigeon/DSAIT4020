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
            df.fillna({col: df[col].mode()[0]}, inplace=True)
        elif col in ['age', 'age_o']:
            df.fillna({col: df[col].median()}, inplace=True)
        elif col in ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
                     'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']:
            df.fillna({col: df[col].mean()}, inplace=True)
        elif col in ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
                     'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts',
                     'music', 'shopping', 'yoga']:
            df.fillna({col: df[col].mean()}, inplace=True)
        elif col in ['like', 'guess_prob_liked', 'expected_happy_with_sd_people',
                     'expected_num_interested_in_me', 'expected_num_matches']:
            df.fillna({col: df[col].mean()}, inplace=True)
        elif col in ['decision', 'match', 'met']:
            df.fillna({col: 0}, inplace=True)
        else:
            # For any other numeric columns, use mean as a general strategy
            df.fillna({col: df[col].mean()}, inplace=True)

    # Display the DataFrame to ensure no missing values remain
    # print(df.isnull().sum())
    return df
