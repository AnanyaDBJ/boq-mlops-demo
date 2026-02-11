import numpy as np
from mlflow.models import make_metric, MetricThreshold
from sklearn.metrics import precision_score, recall_score, f1_score

# Custom metrics to be included. Return empty list if custom metrics are not needed.
# Please refer to custom_metrics parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
def custom_metrics():
    """
    Define custom classification metrics for loan default prediction.
    These metrics complement the built-in metrics provided by MLflow for binary classification.
    """

    def precision_at_threshold_05(eval_df, _builtin_metrics):
        """
        Calculate precision at 0.5 threshold for binary classification.
        Precision = TP / (TP + FP)
        """
        y_true = eval_df["target"]
        y_pred = (eval_df["prediction"] >= 0.5).astype(int)
        return precision_score(y_true, y_pred, zero_division=0)

    def recall_at_threshold_05(eval_df, _builtin_metrics):
        """
        Calculate recall at 0.5 threshold for binary classification.
        Recall = TP / (TP + FN)
        """
        y_true = eval_df["target"]
        y_pred = (eval_df["prediction"] >= 0.5).astype(int)
        return recall_score(y_true, y_pred, zero_division=0)

    def f1_at_threshold_05(eval_df, _builtin_metrics):
        """
        Calculate F1 score at 0.5 threshold for binary classification.
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        y_true = eval_df["target"]
        y_pred = (eval_df["prediction"] >= 0.5).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)

    return [
        make_metric(eval_fn=precision_at_threshold_05, greater_is_better=True, name="precision_at_0.5"),
        make_metric(eval_fn=recall_at_threshold_05, greater_is_better=True, name="recall_at_0.5"),
        make_metric(eval_fn=f1_at_threshold_05, greater_is_better=True, name="f1_at_0.5"),
    ]


# Define model validation rules. Return empty dict if validation rules are not needed.
# Please refer to validation_thresholds parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
def validation_thresholds():
    """
    Define validation thresholds for loan default prediction model.
    Model must meet these criteria to pass validation.
    """
    return {
        # ROC-AUC should be at least 0.70 (70% discrimination ability)
        "roc_auc": MetricThreshold(
            threshold=0.70,
            greater_is_better=True
        ),
        # Precision at 0.5 threshold should be at least 0.65 (65%)
        # This ensures we don't flag too many false positives
        "precision_at_0.5": MetricThreshold(
            threshold=0.65,
            greater_is_better=True
        ),
        # Recall at 0.5 threshold should be at least 0.60 (60%)
        # This ensures we catch at least 60% of actual defaults
        "recall_at_0.5": MetricThreshold(
            threshold=0.60,
            greater_is_better=True
        ),
    }


# Define evaluator config. Return empty dict if validation rules are not needed.
# Please refer to evaluator_config parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
def evaluator_config():
    """
    Evaluator configuration for binary classification.
    Can be used to specify additional evaluation parameters.
    """
    return {
        # Enable SHAP explainability (optional, can be slow for large datasets)
        # "explainability": True,
        # "max_classes_for_multiclass_roc_pr": 2,
    }
