from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
from colorama import Fore, Style

def train_classification_models(X, y, dataset_name):
    print(Fore.YELLOW + f"Training classification models on {dataset_name} dataset" + Style.RESET_ALL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(Fore.CYAN + f"{name} Classification Report:\n" + Style.RESET_ALL, classification_report(y_test, y_pred))

def train_regression_models(X, y, dataset_name):
    print(Fore.YELLOW + f"Training regression models on {dataset_name} dataset" + Style.RESET_ALL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(Fore.CYAN + f"{name} Regression MSE: " + Style.RESET_ALL, mean_squared_error(y_test, y_pred))

def train_clustering_model(X, dataset_name):
    print(Fore.YELLOW + f"Training clustering model on {dataset_name} dataset" + Style.RESET_ALL)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.title(f'{dataset_name} Clustering Results')
    plt.show()