"""
Telecom Churn Prediction with XGBoost and SHAP Explainability.
Stores high-risk customers in TinyDB with reason codes.

Improvements:
1. Model Persistence (save/load)
2. Retention Action Recommendations
3. Cost-Benefit ROI Analysis
4. Customer Risk Timeline
5. Email/Slack Alerts
6. Data Drift Detection
7. Docker Containerization
8. API Authentication
9. Batch Schedule (nightly auto-score)
10. Model Performance Monitoring
"""

import json
import os
import pickle
import smtplib
import warnings
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tinydb import TinyDB, Query

warnings.filterwarnings('ignore')

# --- Improvement #2: Retention Action Recommendations ---
RETENTION_ACTIONS = {
    "Price Shock": "Offer loyalty discount or rate lock for 6 months",
    "Month-to-Month": "Propose 1-year contract with 10% discount",
    "Tech Support Calls": "Assign dedicated support agent and follow-up within 48hrs",
    "Usage Spike": "Offer unlimited data upgrade or waive overage fees",
    "High Monthly Charges": "Offer tier downgrade or bundle discount",
    "Low Tenure": "Provide onboarding bonus and 3-month satisfaction guarantee",
    "No Premium Support": "Offer free 3-month premium support trial",
    "Overage Charges": "Waive current overage and switch to unlimited plan",
    "Electronic Check": "Incentivize switch to auto-pay with $5/month discount",
    "Fiber Optic": "Offer speed upgrade at no extra cost for loyalty",
}


@dataclass
class ChurnPrediction:
    """Data class for a single churn prediction with explanations."""
    customer_id: str
    churn_probability: float
    risk_level: str
    top_reasons: List[str]
    shap_values: Dict[str, float]
    prediction_timestamp: str
    # Improvement #2: Retention actions
    retention_actions: List[str] = field(default_factory=list)
    # Improvement #3: ROI
    expected_revenue_loss: float = 0.0
    retention_cost: float = 0.0
    retention_roi: float = 0.0
    # Improvement #4: Risk timeline
    risk_history: List[Dict] = field(default_factory=list)


class TelecomChurnPredictor:
    """
    XGBoost-based churn predictor with SHAP explainability.
    Stores high-risk customers in TinyDB with all 10 improvements.
    """
    
    # Improvement #1: Model file paths
    MODEL_PATH = 'churn_model.json'
    ENCODERS_PATH = 'encoders.pkl'
    METADATA_PATH = 'model_metadata.json'
    
    def __init__(self, db_path: str = 'at_risk_customers.json',
                 monitoring_db_path: str = 'model_monitoring.json'):
        self.model = None
        self.explainer = None
        self.label_encoders = {}
        self.feature_names = None
        self.db = TinyDB(db_path)
        # Improvement #10: Performance monitoring DB
        self.monitoring_db = TinyDB(monitoring_db_path)
        self.risk_threshold_high = 0.7
        self.risk_threshold_medium = 0.4
        # Improvement #6: Drift detection - store training stats
        self.training_stats = {}
        # Improvement #3: Default retention cost
        self.default_retention_cost = 50.0
        
    # --- Improvement #1: Model Persistence ---
    
    def save_model(self, path: str = None):
        """Save trained model, encoders, and metadata to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = path or self.MODEL_PATH
        
        # Save XGBoost model
        self.model.save_model(model_path)
        
        # Save label encoders
        with open(self.ENCODERS_PATH, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'risk_threshold_high': self.risk_threshold_high,
            'risk_threshold_medium': self.risk_threshold_medium,
            'training_stats': self.training_stats,
            'saved_at': datetime.now().isoformat()
        }
        with open(self.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Encoders saved to {self.ENCODERS_PATH}")
        print(f"Metadata saved to {self.METADATA_PATH}")
    
    def load_model(self, path: str = None) -> bool:
        """
        Load trained model, encoders, and metadata from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = path or self.MODEL_PATH
        
        if not os.path.exists(model_path):
            return False
        
        try:
            # Load XGBoost model
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load label encoders
            if os.path.exists(self.ENCODERS_PATH):
                with open(self.ENCODERS_PATH, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            # Load metadata
            if os.path.exists(self.METADATA_PATH):
                with open(self.METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.risk_threshold_high = metadata.get('risk_threshold_high', 0.7)
                self.risk_threshold_medium = metadata.get('risk_threshold_medium', 0.4)
                self.training_stats = metadata.get('training_stats', {})
            
            # Re-initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            print(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False
    
    def _preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features and prepare data for model."""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'customer_id' in categorical_cols:
            categorical_cols.remove('customer_id')
        if 'churn' in categorical_cols:
            categorical_cols.remove('churn')
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train XGBoost model and initialize SHAP explainer.
        Saves model automatically after training.
        """
        self.feature_names = [col for col in df.columns 
                             if col not in ['customer_id', 'churn']]
        
        df_processed = self._preprocess_features(df, fit=True)
        
        X = df_processed[self.feature_names]
        y = df_processed['churn']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        val_preds = self.model.predict(X_val)
        val_probs = self.model.predict_proba(X_val)[:, 1]
        
        self.explainer = shap.TreeExplainer(self.model)
        
        auc = roc_auc_score(y_val, val_probs)
        
        metrics = {
            'auc': auc,
            'classification_report': classification_report(y_val, val_preds, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, val_preds).tolist(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'trained_at': datetime.now().isoformat()
        }
        
        # Improvement #6: Store training distribution stats for drift detection
        self.training_stats = {
            'mean_tenure': float(df['tenure'].mean()),
            'mean_monthly_charges': float(df['monthly_charges'].mean()),
            'mean_total_charges': float(df['total_charges'].mean()),
            'mean_price_shock': float(df['price_shock_percent'].mean()),
            'mean_tech_support_calls': float(df['tech_support_calls'].mean()),
            'mean_data_usage_gb': float(df['data_usage_gb'].mean()),
            'mean_overage_charges': float(df['overage_charges'].mean()),
            'contract_type_distribution': df['contract_type'].value_counts(normalize=True).to_dict(),
            'churn_rate': float(df['churn'].mean()),
            'sample_size': len(df)
        }
        
        # Improvement #10: Log training metrics for monitoring
        self._log_performance(auc, len(df))
        
        # Improvement #1: Auto-save after training
        self.save_model()
        
        return metrics
    
    # --- Improvement #2: Retention Action Recommendations ---
    
    def _get_retention_actions(self, reasons: List[str]) -> List[str]:
        """Map churn reasons to specific retention actions."""
        actions = []
        for reason in reasons:
            for key, action in RETENTION_ACTIONS.items():
                if key.lower() in reason.lower():
                    if action not in actions:
                        actions.append(action)
                    break
        
        if not actions:
            actions.append("General retention: Schedule customer success check-in call")
        
        return actions[:3]  # Top 3 actions
    
    # --- Improvement #3: Cost-Benefit ROI Analysis ---
    
    def _calculate_retention_roi(self, probability: float, 
                                  monthly_charges: float,
                                  tenure: int) -> Tuple[float, float, float]:
        """
        Calculate retention ROI for a customer.
        
        Returns:
            (expected_revenue_loss, retention_cost, roi)
        """
        # Expected revenue loss if customer churns (12-month projection)
        expected_revenue_loss = monthly_charges * 12 * probability
        
        # Retention cost scales with risk and charges
        if probability >= 0.7:
            retention_cost = monthly_charges * 2  # 2 months free for high risk
        elif probability >= 0.4:
            retention_cost = monthly_charges * 1  # 1 month free for medium
        else:
            retention_cost = 0  # No spend on low risk
        
        # ROI = (expected_loss - retention_cost) / retention_cost
        roi = (expected_revenue_loss - retention_cost) / retention_cost if retention_cost > 0 else 0
        
        return round(expected_revenue_loss, 2), round(retention_cost, 2), round(roi, 2)
    
    # --- Improvement #4: Customer Risk Timeline ---
    
    def _update_risk_timeline(self, customer_id: str, probability: float, 
                               risk_level: str) -> List[Dict]:
        """Update risk history for a customer in TinyDB."""
        Customer = Query()
        existing = self.db.search(Customer.customer_id == customer_id)
        
        history = []
        if existing:
            history = existing[0].get('risk_history', [])
        
        history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'probability': round(probability, 4),
            'risk_level': risk_level
        })
        
        # Keep last 12 entries
        return history[-12:]
    
    # --- Improvement #5: Email/Slack Alerts ---
    
    def send_alert(self, subject: str, body: str, 
                   recipients: List[str] = None,
                   smtp_config: Dict = None):
        """
        Send email alert for high-risk customer detection.
        
        Args:
            subject: Email subject
            body: Email body (HTML supported)
            recipients: List of email addresses
            smtp_config: Dict with host, port, sender, password
        """
        if not recipients or not smtp_config:
            # Log alert instead
            alert_record = {
                'type': 'alert',
                'subject': subject,
                'body': body,
                'timestamp': datetime.now().isoformat(),
                'status': 'logged_only',
                'recipients': recipients or []
            }
            self.monitoring_db.insert(alert_record)
            print(f"[ALERT] {subject}")
            return
        
        try:
            msg = MIMEText(body, 'html')
            msg['Subject'] = subject
            msg['From'] = smtp_config['sender']
            msg['To'] = ', '.join(recipients)
            
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['sender'], smtp_config['password'])
                server.sendmail(smtp_config['sender'], recipients, msg.as_string())
            
            alert_record = {
                'type': 'alert',
                'subject': subject,
                'timestamp': datetime.now().isoformat(),
                'status': 'sent',
                'recipients': recipients
            }
            self.monitoring_db.insert(alert_record)
            print(f"Alert sent to {len(recipients)} recipients")
            
        except Exception as e:
            print(f"Failed to send alert: {e}")
            alert_record = {
                'type': 'alert',
                'subject': subject,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            self.monitoring_db.insert(alert_record)
    
    def check_and_alert(self, predictions: List[ChurnPrediction],
                        threshold: int = 10,
                        recipients: List[str] = None,
                        smtp_config: Dict = None):
        """Send alert if high-risk count exceeds threshold."""
        high_risk_count = len([p for p in predictions if p.risk_level == 'High'])
        
        if high_risk_count >= threshold:
            subject = f"Churn Alert: {high_risk_count} High-Risk Customers Detected"
            body = f"""
            <h2>Telecom Churn Prediction Alert</h2>
            <p><strong>{high_risk_count}</strong> customers have been flagged as high-risk for churn.</p>
            <p>Top reasons observed:</p>
            <ul>
            {''.join(f'<li>{r}</li>' for r in self._get_top_reasons_summary(predictions))}
            </ul>
            <p>Immediate retention action recommended.</p>
            """
            self.send_alert(subject, body, recipients, smtp_config)
    
    def _get_top_reasons_summary(self, predictions: List[ChurnPrediction]) -> List[str]:
        """Get top reason categories from high-risk predictions."""
        high_risk = [p for p in predictions if p.risk_level == 'High']
        reason_counts = {}
        for pred in high_risk:
            for reason in pred.top_reasons:
                category = reason.split(':')[0] if ':' in reason else reason.split('(')[0].strip()
                reason_counts[category] = reason_counts.get(category, 0) + 1
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{r} ({c} customers)" for r, c in sorted_reasons[:5]]
    
    # --- Improvement #6: Data Drift Detection ---
    
    def detect_drift(self, new_data: pd.DataFrame, threshold: float = 0.2) -> Dict:
        """
        Detect data drift by comparing new data distribution to training data.
        
        Args:
            new_data: New incoming data
            threshold: Maximum allowed % shift before alerting
            
        Returns:
            Dict with drift analysis results
        """
        if not self.training_stats:
            return {'error': 'No training stats available. Train model first.'}
        
        drift_results = {}
        alerts = []
        
        numeric_checks = {
            'tenure': ('mean_tenure', new_data['tenure'].mean()),
            'monthly_charges': ('mean_monthly_charges', new_data['monthly_charges'].mean()),
            'total_charges': ('mean_total_charges', new_data['total_charges'].mean()),
            'price_shock_percent': ('mean_price_shock', new_data['price_shock_percent'].mean()),
            'tech_support_calls': ('mean_tech_support_calls', new_data['tech_support_calls'].mean()),
            'data_usage_gb': ('mean_data_usage_gb', new_data['data_usage_gb'].mean()),
            'overage_charges': ('mean_overage_charges', new_data['overage_charges'].mean()),
        }
        
        for feature, (stat_key, new_val) in numeric_checks.items():
            train_val = self.training_stats.get(stat_key, 0)
            if train_val > 0:
                pct_change = abs(new_val - train_val) / train_val
                drifted = pct_change > threshold
                drift_results[feature] = {
                    'training_mean': round(train_val, 2),
                    'new_mean': round(new_val, 2),
                    'pct_change': f"{pct_change:.1%}",
                    'drifted': drifted
                }
                if drifted:
                    alerts.append(f"{feature}: {pct_change:.1%} shift (threshold: {threshold:.0%})")
        
        # Check churn rate shift
        if 'churn' in new_data.columns:
            new_churn_rate = new_data['churn'].mean()
            train_churn_rate = self.training_stats.get('churn_rate', 0)
            churn_shift = abs(new_churn_rate - train_churn_rate) / train_churn_rate if train_churn_rate > 0 else 0
            drift_results['churn_rate'] = {
                'training_rate': f"{train_churn_rate:.1%}",
                'new_rate': f"{new_churn_rate:.1%}",
                'pct_change': f"{churn_shift:.1%}",
                'drifted': churn_shift > threshold
            }
            if churn_shift > threshold:
                alerts.append(f"Churn rate: {churn_shift:.1%} shift")
        
        result = {
            'has_drift': len(alerts) > 0,
            'drift_count': len(alerts),
            'alerts': alerts,
            'details': drift_results,
            'checked_at': datetime.now().isoformat()
        }
        
        # Log drift check
        self.monitoring_db.insert({
            'type': 'drift_check',
            'has_drift': result['has_drift'],
            'drift_count': result['drift_count'],
            'alerts': alerts,
            'checked_at': result['checked_at']
        })
        
        return result
    
    # --- Improvement #10: Model Performance Monitoring ---
    
    def _log_performance(self, auc: float, sample_size: int):
        """Log model performance metrics for monitoring."""
        record = {
            'type': 'performance',
            'auc': round(auc, 4),
            'sample_size': sample_size,
            'measured_at': datetime.now().isoformat()
        }
        self.monitoring_db.insert(record)
    
    def get_performance_history(self) -> List[Dict]:
        """Get all logged performance measurements."""
        Performance = Query()
        records = self.monitoring_db.search(Performance.type == 'performance')
        return sorted(records, key=lambda x: x.get('measured_at', ''))
    
    def check_performance_degradation(self, min_auc: float = 0.60) -> Dict:
        """
        Check if model performance has degraded below threshold.
        
        Returns:
            Dict with degradation analysis
        """
        history = self.get_performance_history()
        
        if not history:
            return {'status': 'no_data', 'message': 'No performance history available'}
        
        latest = history[-1]
        latest_auc = latest.get('auc', 0)
        
        result = {
            'latest_auc': latest_auc,
            'min_auc': min_auc,
            'degraded': latest_auc < min_auc,
            'measured_at': latest.get('measured_at'),
            'history_count': len(history)
        }
        
        if latest_auc < min_auc:
            result['message'] = f"Model AUC ({latest_auc:.3f}) below threshold ({min_auc}). Retraining recommended."
            result['recommendation'] = 'retrain'
        else:
            result['message'] = f"Model AUC ({latest_auc:.3f}) is within acceptable range."
            result['recommendation'] = 'continue'
        
        # Check trend
        if len(history) >= 3:
            recent_aucs = [h.get('auc', 0) for h in history[-3:]]
            result['trend'] = 'declining' if recent_aucs[-1] < recent_aucs[0] else 'stable'
        
        return result
    
    # --- Core prediction methods (enhanced) ---
    
    def _get_feature_importance_names(self, shap_values: np.ndarray, 
                                      feature_values: pd.Series) -> List[str]:
        """Convert SHAP values to human-readable reason codes."""
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:3]
        
        reasons = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_val = shap_values[idx]
            feature_val = feature_values.iloc[idx]
            reason = self._format_reason(feature_name, feature_val, shap_val)
            reasons.append(reason)
        
        return reasons
    
    def _format_reason(self, feature_name: str, value: float, shap_val: float) -> str:
        """Format a feature and its value into a readable reason string."""
        direction = "High" if shap_val > 0 else "Low"
        
        if feature_name in self.label_encoders:
            if int(value) >= 0:
                value = self.label_encoders[feature_name].inverse_transform([int(value)])[0]
                return f"{feature_name.replace('_', ' ').title()}: {value}"
        
        if feature_name == 'price_shock_percent':
            return f"Price Shock: {abs(value):.1f}% {'Increase' if value > 0 else 'Decrease'}"
        elif feature_name == 'monthly_charges':
            return f"{direction} Monthly Charges (${value:.2f})"
        elif feature_name == 'tenure':
            return f"{direction} Tenure ({int(value)} months)"
        elif feature_name == 'tech_support_calls':
            return f"{int(value)} Tech Support Calls (Frustration Indicator)"
        elif feature_name == 'usage_spike_3month':
            return "Usage Spike Detected (Possible Overage)"
        elif feature_name == 'contract_type':
            return f"Contract: {self.label_encoders['contract_type'].inverse_transform([int(value)])[0]}"
        elif feature_name == 'has_premium_support':
            return "No Premium Support" if value == 0 else "Has Premium Support"
        
        return f"{feature_name.replace('_', ' ').title()}: {value}"
    
    def predict_and_explain(self, df: pd.DataFrame) -> List[ChurnPrediction]:
        """
        Generate predictions with SHAP explanations, retention actions, and ROI.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df_processed = self._preprocess_features(df, fit=False)
        X = df_processed[self.feature_names]
        
        probabilities = self.model.predict_proba(X)[:, 1]
        shap_values = self.explainer.shap_values(X)
        
        predictions = []
        timestamp = datetime.now().isoformat()
        
        for i, (idx, row) in enumerate(df.iterrows()):
            prob = probabilities[i]
            
            if prob >= self.risk_threshold_high:
                risk_level = 'High'
            elif prob >= self.risk_threshold_medium:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            if isinstance(shap_values, list):
                instance_shap = shap_values[1][i]
            else:
                instance_shap = shap_values[i]
            
            feature_values = X.iloc[i]
            top_reasons = self._get_feature_importance_names(instance_shap, feature_values)
            
            shap_dict = {
                name: float(val) 
                for name, val in zip(self.feature_names, instance_shap)
            }
            
            # Improvement #2: Retention actions
            retention_actions = self._get_retention_actions(top_reasons)
            
            # Improvement #3: ROI calculation
            monthly_charges = row.get('monthly_charges', 65.0)
            tenure = row.get('tenure', 12)
            expected_loss, ret_cost, roi = self._calculate_retention_roi(
                prob, monthly_charges, tenure
            )
            
            # Improvement #4: Risk timeline
            risk_history = self._update_risk_timeline(row['customer_id'], prob, risk_level)
            
            pred = ChurnPrediction(
                customer_id=row['customer_id'],
                churn_probability=float(prob),
                risk_level=risk_level,
                top_reasons=top_reasons,
                shap_values=shap_dict,
                prediction_timestamp=timestamp,
                retention_actions=retention_actions,
                expected_revenue_loss=expected_loss,
                retention_cost=ret_cost,
                retention_roi=roi,
                risk_history=risk_history
            )
            predictions.append(pred)
        
        return predictions
    
    def store_high_risk(self, predictions: List[ChurnPrediction]) -> int:
        """Store high-risk customers in TinyDB with all enhanced data."""
        high_risk = [p for p in predictions if p.risk_level == 'High']
        
        for pred in high_risk:
            # Convert numpy types to Python native for JSON serialization
            shap_converted = {}
            for k, v in pred.shap_values.items():
                shap_converted[k] = float(v)
            
            risk_history_converted = []
            for h in pred.risk_history:
                risk_history_converted.append({
                    'date': h['date'],
                    'probability': float(h['probability']),
                    'risk_level': h['risk_level']
                })
            
            record = {
                'customer_id': pred.customer_id,
                'churn_probability': float(pred.churn_probability),
                'risk_level': pred.risk_level,
                'top_reasons': pred.top_reasons,
                'shap_values': shap_converted,
                'prediction_timestamp': pred.prediction_timestamp,
                # Enhancement fields
                'retention_actions': pred.retention_actions,
                'expected_revenue_loss': float(pred.expected_revenue_loss),
                'retention_cost': float(pred.retention_cost),
                'retention_roi': float(pred.retention_roi),
                'risk_history': risk_history_converted
            }
            Customer = Query()
            self.db.upsert(record, Customer.customer_id == pred.customer_id)
        
        return len(high_risk)
    
    def export_high_risk_csv(self, predictions: List[ChurnPrediction], 
                             output_path: str = 'high_risk_customers.csv') -> pd.DataFrame:
        """Export high-risk customers to CSV with reasons, actions, and ROI."""
        high_risk = [p for p in predictions if p.risk_level == 'High']
        
        if not high_risk:
            print("No high-risk customers found.")
            return pd.DataFrame()
        
        records = []
        for pred in high_risk:
            record = {
                'customer_id': pred.customer_id,
                'churn_probability': f"{pred.churn_probability:.1%}",
                'risk_level': pred.risk_level,
                'reason_1': pred.top_reasons[0] if len(pred.top_reasons) > 0 else '',
                'reason_2': pred.top_reasons[1] if len(pred.top_reasons) > 1 else '',
                'reason_3': pred.top_reasons[2] if len(pred.top_reasons) > 2 else '',
                'action_1': pred.retention_actions[0] if len(pred.retention_actions) > 0 else '',
                'action_2': pred.retention_actions[1] if len(pred.retention_actions) > 1 else '',
                'action_3': pred.retention_actions[2] if len(pred.retention_actions) > 2 else '',
                'expected_revenue_loss': f"${pred.expected_revenue_loss:.2f}",
                'retention_cost': f"${pred.retention_cost:.2f}",
                'retention_roi': f"{pred.retention_roi:.1f}x",
                'predicted_at': pred.prediction_timestamp
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} high-risk customers to {output_path}")
        return df
    
    def get_global_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def plot_shap_summary(self, df: pd.DataFrame, max_display: int = 10):
        """Generate SHAP summary plot for global interpretability."""
        df_processed = self._preprocess_features(df, fit=False)
        X = df_processed[self.feature_names]
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                         max_display=max_display)


# --- Improvement #9: Batch Schedule (nightly auto-score) ---

def nightly_batch_predict(data_path: str = 'telecom_data.csv',
                          db_path: str = 'at_risk_customers.json',
                          alert_threshold: int = 10,
                          recipients: List[str] = None,
                          smtp_config: Dict = None):
    """
    Run nightly batch prediction on new/updated customer data.
    Designed to be called by a scheduler (cron, APScheduler, etc.)
    """
    print(f"\n=== Nightly Batch: {datetime.now().isoformat()} ===")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} customers")
    except FileNotFoundError:
        print("No data file found. Skipping batch.")
        return
    
    # Initialize predictor and load saved model
    predictor = TelecomChurnPredictor(db_path=db_path)
    
    if not predictor.load_model():
        print("No saved model found. Training new model...")
        metrics = predictor.train(df)
        print(f"New model AUC: {metrics['auc']:.3f}")
    
    # Improvement #6: Check for data drift
    drift_result = predictor.detect_drift(df)
    if drift_result.get('has_drift'):
        print(f"⚠️ Data drift detected: {drift_result['alerts']}")
    
    # Improvement #10: Check performance degradation
    perf_check = predictor.check_performance_degradation()
    if perf_check.get('degraded'):
        print(f"⚠️ Model degraded: {perf_check['message']}")
        print("Retraining model...")
        metrics = predictor.train(df)
        print(f"Retrained AUC: {metrics['auc']:.3f}")
    
    # Run predictions
    predictions = predictor.predict_and_explain(df)
    
    # Store high-risk
    stored_count = predictor.store_high_risk(predictions)
    print(f"Stored {stored_count} high-risk customers")
    
    # Improvement #5: Check alert threshold
    predictor.check_and_alert(predictions, threshold=alert_threshold,
                              recipients=recipients, smtp_config=smtp_config)
    
    # Export CSV
    predictor.export_high_risk_csv(predictions)
    
    print(f"=== Batch Complete ===\n")
    return predictions


def main():
    """Main execution pipeline."""
    try:
        df = pd.read_csv('telecom_data.csv')
        print(f"Loaded existing dataset: {df.shape}")
    except FileNotFoundError:
        from data_generator import generate_dataset
        df = generate_dataset(n_samples=5000, output_path='telecom_data.csv')
    
    predictor = TelecomChurnPredictor(db_path='at_risk_customers.json')
    
    # Improvement #1: Try loading saved model first
    if not predictor.load_model():
        print("No saved model found. Training new model...")
        metrics = predictor.train(df, test_size=0.2)
        print(f"Validation AUC: {metrics['auc']:.3f}")
    else:
        print("Using saved model.")
    
    # Global feature importance
    print("\nGlobal Feature Importance:")
    importance_df = predictor.get_global_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # Improvement #6: Drift detection
    print("\nDrift Detection:")
    drift_result = predictor.detect_drift(df)
    print(f"  Drift detected: {drift_result.get('has_drift', False)}")
    if drift_result.get('alerts'):
        for alert in drift_result['alerts']:
            print(f"  ⚠️ {alert}")
    
    # Improvement #10: Performance check
    print("\nPerformance Monitoring:")
    perf = predictor.check_performance_degradation()
    print(f"  {perf['message']}")
    
    # Generate predictions
    print("\nGenerating predictions with SHAP explanations...")
    predictions = predictor.predict_and_explain(df)
    
    # Store high-risk
    stored_count = predictor.store_high_risk(predictions)
    print(f"Stored {stored_count} high-risk customers in TinyDB")
    
    # Improvement #5: Alert check
    predictor.check_and_alert(predictions, threshold=10)
    
    # Export CSV
    high_risk_df = predictor.export_high_risk_csv(predictions, 
                                                   output_path='high_risk_customers.csv')
    
    # Summary
    risk_counts = pd.Series([p.risk_level for p in predictions]).value_counts()
    print("\nRisk Distribution:")
    print(risk_counts)
    
    # Improvement #3: ROI Summary
    high_risk_preds = [p for p in predictions if p.risk_level == 'High']
    if high_risk_preds:
        total_loss = sum(p.expected_revenue_loss for p in high_risk_preds)
        total_cost = sum(p.retention_cost for p in high_risk_preds)
        avg_roi = np.mean([p.retention_roi for p in high_risk_preds if p.retention_roi > 0])
        print(f"\nROI Summary (High-Risk):")
        print(f"  Total Expected Revenue Loss: ${total_loss:,.2f}")
        print(f"  Total Retention Investment: ${total_cost:,.2f}")
        print(f"  Average Retention ROI: {avg_roi:.1f}x")
    
    print("\nSample High-Risk Customer:")
    if high_risk_preds:
        sample = high_risk_preds[0]
        print(f"  Customer: {sample.customer_id}")
        print(f"  Probability: {sample.churn_probability:.1%}")
        print(f"  Top Reasons:")
        for i, r in enumerate(sample.top_reasons[:3], 1):
            print(f"    {i}. {r}")
        print(f"  Retention Actions:")
        for i, a in enumerate(sample.retention_actions[:3], 1):
            print(f"    {i}. {a}")
        print(f"  Expected Revenue Loss: ${sample.expected_revenue_loss:.2f}")
        print(f"  Retention Cost: ${sample.retention_cost:.2f}")
        print(f"  ROI: {sample.retention_roi:.1f}x")
    
    return predictor, predictions, high_risk_df


if __name__ == '__main__':
    predictor, predictions, high_risk_df = main()
