from src.data_preparation import load_data, preprocess_data
from src.pca_analysis import apply_pca
from src.model_training import train_and_evaluate


def main():
    # Load and preprocess data
    X, y = load_data()
    X, y = preprocess_data(X, y)

    # Apply PCA
    X_pca, components = apply_pca(X, n_components=5)

    # Train and evaluate model
    best_params, best_accuracy = train_and_evaluate(X_pca, y)

    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
