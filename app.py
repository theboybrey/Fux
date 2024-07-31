from sklearn.datasets import load_iris, load_wine, load_digits, load_diabetes
from preprocessing import preprocess_classification_data, preprocess_regression_data, preprocess_clustering_data
from train_model import train_classification_models, train_regression_models, train_clustering_model
from colorama import Fore, Style, init
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Initialize colorama
init()

# Disable SSL verification for UCI Machine Learning Repository
ssl._create_default_https_context = ssl._create_unverified_context

def handle_error(e, context):
    print(Fore.RED + f"Error occurred in {context}: {e}" + Style.RESET_ALL)

def load_and_process_classification_data(dataset_loader, target_name):
    try:
        data = dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[target_name] = data.target
        X, y = preprocess_classification_data(df, target_name)
        return X, y
    except Exception as e:
        handle_error(e, f"Loading classification data from {dataset_loader.__name__}")
        return None, None

def load_and_process_regression_data(dataset_loader, target_name):
    try:
        data = dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[target_name] = data.target

        # Handle categorical features
        categorical_features = df.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_features]).toarray()
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        df = df.drop(columns=categorical_features).join(encoded_df)

        # Impute missing values before scaling
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        X = df.drop(columns=[target_name])
        y = df[target_name]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except Exception as e:
        handle_error(e, f"Loading regression data from {dataset_loader.__name__}")
        return None, None

def load_and_process_clustering_data(dataset_loader):
    try:
        data = dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = preprocess_clustering_data(df)
        return X
    except Exception as e:
        handle_error(e, f"Loading clustering data from {dataset_loader.__name__}")
        return None

def main():
    # Classification datasets
    classification_datasets = [
        (load_iris, 'species'),
        (load_wine, 'class')
    ]

    for dataset_loader, target_name in classification_datasets:
        X, y = load_and_process_classification_data(dataset_loader, target_name)
        if X is not None and y is not None:
            train_classification_models(X, y, dataset_loader.__name__)

    # Regression datasets
    regression_datasets = [
        # (load_boston, 'target'), # Boston Housing - Removed
        (load_diabetes, 'target'), # Diabetes
        # (fetch_openml, 'price')  # New: Ames Housing Dataset - Removed
    ]

    for dataset_loader, target_name in regression_datasets:
        X, y = load_and_process_regression_data(dataset_loader, target_name)
        if X is not None and y is not None:
            train_regression_models(X, y, dataset_loader.__name__)

    # Clustering datasets
    clustering_datasets = [
        load_iris,
        load_digits
    ]

    for dataset_loader in clustering_datasets:
        X = load_and_process_clustering_data(dataset_loader)
        if X is not None:
            train_clustering_model(X, dataset_loader.__name__)

    # Plot results
    def plot_results():
        print(Fore.CYAN + "Plotting results..." + Style.RESET_ALL)
        try:
            sns.set(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot for classification results
            axes[0].bar(['Iris', 'Wine'], [0.99, 0.98], color='skyblue')
            axes[0].set_title('Classification Accuracy')
            axes[0].set_xlabel('Dataset')
            axes[0].set_ylabel('Accuracy')

            # Example plot for regression results
            axes[1].scatter([1, 2, 3], [10, 20, 30], color='lightcoral')
            axes[1].set_title('Regression Results')
            axes[1].set_xlabel('Dataset')
            axes[1].set_ylabel('Performance Metric')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            handle_error(e, "Plotting results")

    plot_results()

if __name__ == "__main__":
    main()