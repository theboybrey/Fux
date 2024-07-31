import pandas as pd
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style

def load_dataset(file_path, columns=None, delimiter=','):
    print(Fore.GREEN + f"Loading dataset from {file_path}" + Style.RESET_ALL)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, delimiter=delimiter)
    elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')  # Specify engine for xlsx files
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    if columns:
        df.columns = columns
    return df

def preprocess_regression_data(df, target_column):
    print(Fore.BLUE + "Preprocessing regression data" + Style.RESET_ALL)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, forcing non-numeric to NaN
    df = df.dropna()  # Drop rows with NaN values
    if target_column not in df.columns:
        raise KeyError(f"Column {target_column} not found in the dataset.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def preprocess_clustering_data(df, columns_to_drop=None):
    print(Fore.BLUE + "Preprocessing clustering data" + Style.RESET_ALL)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, forcing non-numeric to NaN
    df = df.dropna()  # Drop rows with NaN values
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

def preprocess_classification_data(df, target_column):
    print(Fore.BLUE + "Preprocessing classification data" + Style.RESET_ALL)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, forcing non-numeric to NaN
    df = df.dropna()  # Drop rows with NaN values
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y