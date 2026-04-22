"""
Dashboard for visualizing churn predictions, SHAP explanations,
retention ROI, drift detection, and model monitoring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tinydb import TinyDB, Query


def show_risk_distribution(predictions_df: pd.DataFrame):
    """Plot risk level distribution."""
    plt.figure(figsize=(8, 5))
    risk_counts = predictions_df['risk_level'].value_counts()
    colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
    
    risk_counts.plot(kind='bar', color=[colors.get(x, '#3498db') for x in risk_counts.index])
    plt.title('Customer Risk Distribution')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=150)
    plt.show()


def show_feature_importance(importance_df: pd.DataFrame, top_n: int = 10):
    """Plot global feature importance."""
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()


def show_roi_analysis(predictions):
    """Plot cost-benefit ROI analysis for high-risk customers."""
    high_risk = [p for p in predictions if p.risk_level == 'High']
    if not high_risk:
        print("No high-risk customers for ROI analysis.")
        return
    
    losses = [float(p.expected_revenue_loss) for p in high_risk]
    costs = [float(p.retention_cost) for p in high_risk]
    rois = [float(p.retention_roi) for p in high_risk]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Revenue loss vs retention cost
    axes[0].hist(losses, bins=30, alpha=0.7, color='#e74c3c', label='Revenue at Risk')
    axes[0].hist(costs, bins=30, alpha=0.7, color='#f39c12', label='Retention Cost')
    axes[0].set_title('Revenue Loss vs Retention Cost')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    # ROI distribution
    axes[1].hist(rois, bins=30, color='#27ae60', alpha=0.7)
    axes[1].axvline(x=np.mean(rois), color='#2c3e50', linestyle='--', label=f'Mean: {np.mean(rois):.1f}x')
    axes[1].set_title('Retention ROI Distribution')
    axes[1].set_xlabel('ROI (x)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    # Summary totals
    total_loss = sum(losses)
    total_cost = sum(costs)
    net_savings = total_loss - total_cost
    categories = ['Revenue\nat Risk', 'Retention\nInvestment', 'Net Savings\nPotential']
    values = [total_loss, total_cost, net_savings]
    bar_colors = ['#e74c3c', '#f39c12', '#27ae60']
    axes[2].bar(categories, values, color=bar_colors)
    axes[2].set_title('Aggregate Cost-Benefit ($)')
    axes[2].set_ylabel('Amount ($)')
    
    plt.tight_layout()
    plt.savefig('roi_analysis.png', dpi=150)
    plt.show()


def show_retention_actions(predictions, top_n: int = 10):
    """Plot most common retention action recommendations."""
    from collections import Counter
    all_actions = []
    for p in predictions:
        if p.risk_level == 'High' and hasattr(p, 'retention_actions'):
            all_actions.extend(p.retention_actions)
    
    if not all_actions:
        print("No retention actions to display.")
        return
    
    action_counts = Counter(all_actions).most_common(top_n)
    actions, counts = zip(*action_counts)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(actions)), counts, color='#27ae60')
    plt.yticks(range(len(actions)), [a[:50] for a in actions])
    plt.xlabel('Frequency')
    plt.title('Top Retention Action Recommendations')
    plt.tight_layout()
    plt.savefig('retention_actions.png', dpi=150)
    plt.show()


def show_drift_results(drift_result: dict):
    """Visualize drift detection results."""
    if not drift_result or 'details' not in drift_result:
        print("No drift data to visualize.")
        return
    
    details = drift_result['details']
    features = list(details.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    numeric_features = [f for f in features if 'pct_change' in details[f]]
    if not numeric_features:
        print("No numeric drift data to plot.")
        return
    
    pct_changes = [float(details[f].get('pct_change', 0).replace('%', '')) for f in numeric_features]
    drifted = [details[f].get('drifted', False) for f in numeric_features]
    colors = ['#e74c3c' if d else '#27ae60' for d in drifted]
    
    ax.barh(range(len(numeric_features)), pct_changes, color=colors)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_yticks(range(len(numeric_features)))
    ax.set_yticklabels([f.replace('_', ' ') for f in numeric_features])
    ax.set_xlabel('% Change from Training Baseline')
    ax.set_title('Data Drift Detection by Feature')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Drifted'), Patch(facecolor='#27ae60', label='Stable')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('drift_detection.png', dpi=150)
    plt.show()


def show_performance_history(predictor):
    """Plot model performance history from monitoring DB."""
    history = predictor.get_performance_history()
    if not history:
        print("No performance history available.")
        return
    
    dates = [h.get('measured_at', '')[:10] for h in history]
    aucs = [h.get('auc', 0) for h in history]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, aucs, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax.axhline(y=0.65, color='#e74c3c', linestyle='--', alpha=0.7, label='Degradation Threshold')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Model AUC Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('AUC')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_history.png', dpi=150)
    plt.show()


def query_at_risk_db(db_path: str = 'at_risk_customers.json'):
    """Query and display at-risk customers from TinyDB with enhanced fields."""
    db = TinyDB(db_path)
    records = db.all()
    
    if not records:
        print("No at-risk customers in database.")
        return None
    
    df = pd.DataFrame(records)
    print(f"\nAt-Risk Customers in Database: {len(df)}")
    
    display_cols = ['customer_id', 'churn_probability', 'risk_level', 'top_reasons']
    if 'retention_actions' in df.columns:
        display_cols.append('retention_actions')
    if 'expected_revenue_loss' in df.columns:
        display_cols.append('expected_revenue_loss')
    if 'retention_roi' in df.columns:
        display_cols.append('retention_roi')
    
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head())
    return df


if __name__ == '__main__':
    from train_and_explain import main
    
    predictor, predictions, high_risk_df = main()
    
    # Convert predictions to DataFrame for visualization
    pred_df = pd.DataFrame([
        {
            'customer_id': p.customer_id,
            'churn_probability': p.churn_probability,
            'risk_level': p.risk_level
        }
        for p in predictions
    ])
    
    # Standard visualizations
    show_risk_distribution(pred_df)
    show_feature_importance(predictor.get_global_feature_importance())
    
    # Enhanced visualizations
    show_roi_analysis(predictions)
    show_retention_actions(predictions)
    
    # Drift detection
    df = pd.read_csv('telecom_data.csv')
    drift_result = predictor.detect_drift(df)
    show_drift_results(drift_result)
    
    # Performance monitoring
    show_performance_history(predictor)
    
    # Query database
    query_at_risk_db()
