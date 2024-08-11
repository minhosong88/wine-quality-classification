from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_pca(X, n_components=5):
    """Apply PCA to reduce dimensionality."""
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(scaled_X)
    return X_pca, pca.components_
