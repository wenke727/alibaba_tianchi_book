import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def null_importance_feature_selection(X, y, feature_names, n_estimators=100, random_state=42):
    """
    Performs null importance feature selection to identify useful features in a dataset.
    This method involves comparing the importance of features with their importance when
    the target variable is shuffled. If the importance of a feature with the original target
    is not significantly higher than its importance with the shuffled target, the feature 
    may be considered as not useful.

    Parameters:
        - X: Features dataset.
        - y: Target variable.
        - feature_names: List of feature names.
        - n_estimators: Number of trees in the forest. Default is 100.
        - random_state: Random state for reproducibility. Default is 42.

    Returns:
        - A DataFrame with feature names, their original importance, shuffled importance,
          and a boolean indicating whether the feature is considered useful or not.
        - A list of feature names that are considered useful.
    """

    def calculate_feature_importance(X, y):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)
        return model.feature_importances_

    original_importance = calculate_feature_importance(X, y)

    shuffled_y = np.random.permutation(y)
    shuffled_importance = calculate_feature_importance(X, shuffled_y)

    feature_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Original Importance': original_importance,
        'Shuffled Importance': shuffled_importance
    })

    feature_comparison['Useful'] = feature_comparison['Original Importance'] > feature_comparison['Shuffled Importance']
    useful_feature_names = feature_comparison[feature_comparison.Useful].Feature.values.tolist()
    return feature_comparison, useful_feature_names


if __name__ == "__main__":
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Perform null importance feature selection
    feature_importance_analysis, useful_feature_names = null_importance_feature_selection(X, y, iris.feature_names)
    print(f"useful_feature_names: {useful_feature_names}")
    print(feature_importance_analysis)
