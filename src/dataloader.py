import pandas as pd
from scipy.io import arff


def load_arff_to_dataframe(file_path):
    """
    Load a .arff file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the .arff file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the .arff file.
    """
    # Read the .arff file
    data, meta = arff.loadarff(file_path)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)

    # Decode any byte-string columns to normal strings if necessary
    for col in df.select_dtypes(["object"]):
        df[col] = df[col].str.decode('utf-8')

    return df
