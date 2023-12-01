import math
import pandas as pd
import matplotlib.pyplot as plt

def add_predictions_and_labels(clf, X, y):
    """
    Adds predictions and actual labels to the dataset.

    Args:
        clf: Trained classifier.
        X: DataFrame containing data to predict.
        y: Series containing actual labels.

    Returns:
        DataFrame with added prediction and ground truth columns.
    """
    X_copy = X.copy()
    X_copy['pred_label'] = clf.predict(X)
    X_copy['gt_label'] = y

    return X_copy

def analyze_and_visualize_conditions(X, scaler, gt, pred, wo_pred_gt, n_col=4):
    """
    Analyzes and visualizes data for specific ground truth and prediction conditions.
    可视化混淆矩阵

    Args:
        X: DataFrame containing data with predictions and labels.
        scaler: Pre-fitted scaler (e.g., StandardScaler).
        useful_feature_names: List of feature names for plotting.
        gt: Ground truth value for filtering.
        pred: Prediction value for filtering.
        wo_pred_gt: Exclude records where prediction equals ground truth.

    Returns:
        DataFrame containing the filtered and labeled data.
    """
    X_scaled = X.copy()
    label_cols = ['pred', 'gt']
    cols = X.columns.difference(label_cols)
    X_scaled.loc[:, cols] = scaler.inverse_transform(X[cols])
    X_scaled.loc[:, label_cols] = X[label_cols]

    conditions = [
        (gt, lambda x: x.query(f'gt_label == pred_label == @gt')),
        (pred, lambda x: x.query('gt_label == pred_label == @pred') if not wo_pred_gt else pd.DataFrame()),
        (f'{gt}->{pred}', lambda x: x.query('gt_label == @gt and pred_label == @pred'))
    ]

    filtered_data = []
    for label, condition_func in conditions:
        subset = condition_func(X_scaled)
        if not subset.empty:
            subset['label'] = label
            filtered_data.append(subset)

    result_X = pd.concat(filtered_data)
    plot_kde_grid(result_X, n_col=n_col, hue='label', common_norm=False)

    return filtered_data


def plot_kde_grid(df, n_col, hue="label", n_row=None, *args, **kwargs):
    """
    对DataFrame中的数值列绘制核密度估计图。

    :param df: pandas DataFrame，包含数值型数据。
    :param n_col: 网格的列数。
    :param n_row: 网格的行数。如果为None，则会自动计算。
    :param hue: 用于在kdeplot中进行分组的列名，默认为'label'。
    """
    num_cols = sorted(df.select_dtypes(include=["number"]).columns)
    logger.debug(
        f"plot kde plot for: {num_cols}, except {[i for i in df.columns if i not in num_cols]}"
    )

    if n_row is None:
        n_row = math.ceil(len(num_cols) / n_col)

    # create axes
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 3))
    if n_row * n_col > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(num_cols):
        if i < n_row * n_col:
            sns.kdeplot(
                data=df,
                x=col,
                fill=True,
                hue=df[hue] if hue in df.columns else None,
                ax=axes[i],
                *args,
                **kwargs,
            )
            axes[i].set_title(col)
            axes[i].set_xlabel("")
        else:
            break

    # 隐藏多余的图表
    for j in range(i + 1, n_row * n_col):
        axes[j].axis("off")

    plt.tight_layout()
    # plt.show()

    return fig

