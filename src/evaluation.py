"""
evaluation.py
=============
Evaluation Pipeline for Air Pollution Risk Classification
Project: Classifying Air Pollution Risk Levels in Makkah
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import pandas as pd


def calculate_metrics(y_true, y_pred, labels=None):
    """
    Computes weighted Accuracy, Precision, Recall, and F1-score.
    Weighted averaging handles class imbalance in pollution risk levels.
    """
    if labels is None:
        labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }


def print_classification_report(y_true, y_pred, labels=None):
    """
    Prints per-class precision, recall, and F1-score for detailed analysis.
    """
    
    # الحصول على الفئات الفعلية من البيانات
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=unique_labels,
        digits=4
    )
    print(report)
    
    return report


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Visualizes prediction accuracy via a confusion matrix heatmap.
    Diagonal elements represent correct classifications.
    """
    
    # الحصول على الفئات الفعلية من البيانات
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='white'
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Count')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.xticks(range(len(unique_labels)), unique_labels)
        plt.yticks(range(len(unique_labels)), unique_labels)
    
    plt.title('Confusion Matrix - Air Pollution Risk Classification', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved to: {save_path}")
    
    plt.show()
    
    return cm


def save_results(metrics, filepath='outputs/evaluation_results.csv'):
    """
    Exports evaluation metrics to CSV for documentation and reproducibility.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    results_df = pd.DataFrame([metrics])
    results_df['timestamp'] = pd.Timestamp.now()
    results_df['model'] = 'Decision Tree'
    results_df['dataset'] = 'Makkah Pollution 2024'
    
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    
    return results_df


def full_evaluation(y_true, y_pred, labels=None, save_path=None):
    """
    Executes complete evaluation pipeline: metrics, report, visualization, and export.
    """
    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("Project: Makkah Air Pollution Risk Classification")
    print("=" * 60)
    
    # Step 1: Metrics
    print("\n[1/4] Computing metrics...")
    metrics = calculate_metrics(y_true, y_pred, labels)
    print(f"   Accuracy:  {metrics['accuracy']}")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall:    {metrics['recall']}")
    print(f"   F1-Score:  {metrics['f1_score']}")
    
    # Step 2: Report
    print("\n[2/4] Generating classification report...")
    print_classification_report(y_true, y_pred, labels)
    
    # Step 3: Visualization
    print("\n[3/4] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, labels, save_path)
    
    # Step 4: Export
    print("\n[4/4] Saving results...")
    save_results(metrics)
    
    print("\n" + "=" * 60)
    print("Evaluation completed successfully.")
    print("=" * 60)
    
    return metrics


# Local testing block
if __name__ == "__main__":
    
    print("=" * 60)
    print("TESTING evaluation.py")
    print("=" * 60)
    
    y_true_test = [
        'High', 'Medium', 'Low', 'High', 'Medium',
        'Low', 'High', 'Medium', 'Low', 'High',
        'Medium', 'Low', 'High', 'Medium', 'Low'
    ]
    
    y_pred_test = [
        'High', 'Medium', 'Low', 'High', 'Medium',
        'Low', 'Medium', 'Medium', 'Low', 'High',
        'Medium', 'Low', 'High', 'Low', 'Low'
    ]
    
    full_evaluation(y_true_test, y_pred_test, save_path='test_confusion_matrix.png')
    
    print("\nevaluation.py is ready for integration.")


def evaluate_performance(model, X_test, y_test, labels=None):
    """
    Legacy function for backward compatibility with main.py.
    Computes predictions and calls full_evaluation.
    """
    if labels is None:
        labels = ['Low', 'Medium', 'High']
    
    y_pred = model.predict(X_test)
    
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE METRICS")
    print("=" * 60)
    
    return full_evaluation(y_test, y_pred, labels, save_path='confusion_matrix.png')