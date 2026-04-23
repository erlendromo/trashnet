import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_feature_vectors(train_features, val_features=None, test_features=None):
    """
    Scale feature vectors using StandardScaler.

    IMPORTANT:
    - Fit scaler ONLY on training data
    - Transform validation and test data using the same scaler

    This prevents data leakage and is the correct ML practice.

    Parameters
    ----------
    train_features : list or np.ndarray
        Shape: (n_samples_train, n_features)

    val_features : list or np.ndarray, optional
        Shape: (n_samples_val, n_features)

    test_features : list or np.ndarray, optional
        Shape: (n_samples_test, n_features)

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training features

    X_val_scaled : np.ndarray or None
        Scaled validation features

    X_test_scaled : np.ndarray or None
        Scaled test features

    scaler : StandardScaler
        Fitted scaler object
        (useful for saving/loading later)
    """

    # Convert to numpy arrays
    X_train = np.array(train_features)

    if val_features is not None:
        X_val = np.array(val_features)
    else:
        X_val = None

    if test_features is not None:
        X_test = np.array(test_features)
    else:
        X_test = None

    print("Scaling feature vectors...\n")

    # Create scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and test using same scaler
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )
