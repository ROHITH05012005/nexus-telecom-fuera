"""
Flask web dashboard for Telecom Churn Prediction System.
Displays high-risk customers with SHAP explainability.

Improvements integrated:
1. Model Persistence (load saved model, no retrain on startup)
2. Retention Actions in API responses
3. Cost-Benefit ROI in API responses
4. Risk Timeline in customer details
5. Alert API endpoint
6. Data Drift Detection API
7. Docker support (Dockerfile + docker-compose)
8. API Authentication
9. Scheduler integration
10. Performance Monitoring API
"""

import os
import pickle
import uuid
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename
from tinydb import TinyDB, Query

app = Flask(__name__)

# --- Improvement #8: API Authentication ---
API_KEY = os.environ.get('API_KEY', 'changeme')
AUTH_ENABLED = os.environ.get('AUTH_ENABLED', 'false').lower() == 'true'


def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        
        provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if provided_key == API_KEY:
            return f(*args, **kwargs)
        
        return jsonify({'error': 'Unauthorized. Provide X-API-Key header.'}), 401
    return decorated


# Initialize database connection
db = TinyDB('at_risk_customers.json')

# --- Improvement #1: Load saved model instead of retraining ---
try:
    from train_and_explain import TelecomChurnPredictor
    predictor = TelecomChurnPredictor()
    
    # Try loading saved model first
    if not predictor.load_model():
        print("No saved model found. Training new model...")
        df = pd.read_csv('telecom_data.csv')
        predictor.train(df)
    print("Model ready")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


def load_customer_data(risk_level=None, search_query=None):
    """Load customers from TinyDB with optional filtering."""
    Customer = Query()
    
    if risk_level and risk_level != 'All':
        records = db.search(Customer.risk_level == risk_level)
    else:
        records = db.all()
    
    if search_query:
        records = [r for r in records if search_query.lower() in r.get('customer_id', '').lower()]
    
    # Sort by churn probability descending
    records = sorted(records, key=lambda x: x.get('churn_probability', 0), reverse=True)
    return records


def get_statistics():
    """Calculate summary statistics."""
    all_records = db.all()
    
    if not all_records:
        return {}
    
    total = len(all_records)
    high_risk = len([r for r in all_records if r.get('risk_level') == 'High'])
    medium_risk = len([r for r in all_records if r.get('risk_level') == 'Medium'])
    low_risk = len([r for r in all_records if r.get('risk_level') == 'Low'])
    
    avg_probability = sum(r.get('churn_probability', 0) for r in all_records) / total
    
    return {
        'total_at_risk': total,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'avg_probability': f"{avg_probability:.1%}",
        'high_risk_percent': f"{high_risk / total * 100:.1f}%"
    }


def get_top_reasons_analysis(limit=10):
    """Analyze most common churn reasons across all customers."""
    all_records = db.all()
    reason_counts = {}
    
    for record in all_records:
        for reason in record.get('top_reasons', []):
            # Extract main category (before colon if present)
            category = reason.split(':')[0] if ':' in reason else reason
            reason_counts[category] = reason_counts.get(category, 0) + 1
    
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_reasons[:limit]


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/customers')
def get_customers():
    """API endpoint for customer data."""
    risk_filter = request.args.get('risk_level', 'All')
    search = request.args.get('search', '')
    
    customers = load_customer_data(risk_filter, search)
    
    # Format for display
    formatted = []
    for c in customers[:500]:  # Limit to 500 for performance
        formatted.append({
            'customer_id': c.get('customer_id'),
            'churn_probability': f"{c.get('churn_probability', 0):.1%}",
            'risk_level': c.get('risk_level'),
            'risk_class': c.get('risk_level', '').lower().replace('-', ''),
            'reasons': c.get('top_reasons', []),
            'predicted_at': c.get('prediction_timestamp', '').replace('T', ' ')[:19]
        })
    
    return jsonify(formatted)


@app.route('/api/statistics')
def get_stats():
    """API endpoint for dashboard statistics."""
    return jsonify(get_statistics())


@app.route('/api/reasons')
def get_reasons():
    """API endpoint for top churn reasons."""
    return jsonify(get_top_reasons_analysis())


@app.route('/api/customer/<customer_id>')
def get_customer_detail(customer_id):
    """Get detailed SHAP values, retention actions, ROI, and risk timeline."""
    Customer = Query()
    record = db.search(Customer.customer_id == customer_id)
    
    if not record:
        return jsonify({'error': 'Customer not found'}), 404
    
    record = record[0]
    
    # Sort SHAP values by absolute magnitude
    shap_values = record.get('shap_values', {})
    sorted_shap = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    return jsonify({
        'customer_id': record.get('customer_id'),
        'churn_probability': record.get('churn_probability'),
        'risk_level': record.get('risk_level'),
        'top_reasons': record.get('top_reasons'),
        'shap_breakdown': [
            {'feature': k, 'impact': round(v, 4), 'direction': 'increases' if v > 0 else 'decreases'}
            for k, v in sorted_shap
        ],
        # Improvement #2: Retention actions
        'retention_actions': record.get('retention_actions', []),
        # Improvement #3: ROI
        'expected_revenue_loss': record.get('expected_revenue_loss', 0),
        'retention_cost': record.get('retention_cost', 0),
        'retention_roi': record.get('retention_roi', 0),
        # Improvement #4: Risk timeline
        'risk_history': record.get('risk_history', [])
    })


@app.route('/api/predict', methods=['POST'])
def predict_customer():
    """
    Predict churn for a new customer.
    Accepts JSON with customer features.
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Generate customer ID if not provided
        customer_id = data.get('customer_id', f'NEW_{uuid.uuid4().hex[:8].upper()}')
        
        # Create DataFrame from input
        input_df = pd.DataFrame([{
            'customer_id': customer_id,
            'tenure': int(data.get('tenure', 12)),
            'monthly_charges': float(data.get('monthly_charges', 65.0)),
            'total_charges': float(data.get('total_charges', 780.0)),
            'contract_type': data.get('contract_type', 'Month-to-Month'),
            'payment_method': data.get('payment_method', 'Electronic Check'),
            'internet_service': data.get('internet_service', 'Fiber Optic'),
            'tech_support_calls': int(data.get('tech_support_calls', 0)),
            'has_premium_support': int(data.get('has_premium_support', 0)),
            'price_shock_percent': float(data.get('price_shock_percent', 0)),
            'usage_spike_3month': int(data.get('usage_spike_3month', 0)),
            'data_usage_gb': float(data.get('data_usage_gb', 50)),
            'overage_charges': float(data.get('overage_charges', 0))
        }])
        
        # Get prediction
        predictions = predictor.predict_and_explain(input_df)
        pred = predictions[0]
        
        # Prepare response
        result = {
            'customer_id': pred.customer_id,
            'churn_probability': pred.churn_probability,
            'risk_level': pred.risk_level,
            'top_reasons': pred.top_reasons,
            'churn_prediction': 1 if pred.churn_probability >= 0.5 else 0,
            # Improvement #2: Retention actions
            'retention_actions': pred.retention_actions,
            # Improvement #3: ROI
            'expected_revenue_loss': pred.expected_revenue_loss,
            'retention_cost': pred.retention_cost,
            'retention_roi': pred.retention_roi
        }
        
        # If high risk, optionally store in DB
        if data.get('store_result', False) and pred.risk_level == 'High':
            record = {
                'customer_id': pred.customer_id,
                'churn_probability': pred.churn_probability,
                'risk_level': pred.risk_level,
                'top_reasons': pred.top_reasons,
                'shap_values': pred.shap_values,
                'prediction_timestamp': pred.prediction_timestamp,
                'input_data': data
            }
            Customer = Query()
            db.upsert(record, Customer.customer_id == pred.customer_id)
            result['stored_in_db'] = True
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers.
    Accepts JSON array of customer features.
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Expected array of customers'}), 400
    
    try:
        # Create DataFrame from inputs
        rows = []
        for i, item in enumerate(data):
            rows.append({
                'customer_id': item.get('customer_id', f'BATCH_{i}_{uuid.uuid4().hex[:6].upper()}'),
                'tenure': int(item.get('tenure', 12)),
                'monthly_charges': float(item.get('monthly_charges', 65.0)),
                'total_charges': float(item.get('total_charges', 780.0)),
                'contract_type': item.get('contract_type', 'Month-to-Month'),
                'payment_method': item.get('payment_method', 'Electronic Check'),
                'internet_service': item.get('internet_service', 'Fiber Optic'),
                'tech_support_calls': int(item.get('tech_support_calls', 0)),
                'has_premium_support': int(item.get('has_premium_support', 0)),
                'price_shock_percent': float(item.get('price_shock_percent', 0)),
                'usage_spike_3month': int(item.get('usage_spike_3month', 0)),
                'data_usage_gb': float(item.get('data_usage_gb', 50)),
                'overage_charges': float(item.get('overage_charges', 0))
            })
        
        input_df = pd.DataFrame(rows)
        
        # Get predictions
        predictions = predictor.predict_and_explain(input_df)
        
        # Format results
        results = []
        for pred in predictions:
            results.append({
                'customer_id': pred.customer_id,
                'churn_probability': pred.churn_probability,
                'risk_level': pred.risk_level,
                'top_reasons': pred.top_reasons,
                'churn_prediction': 1 if pred.churn_probability >= 0.5 else 0
            })
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'high_risk': len([r for r in results if r['risk_level'] == 'High'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/excel', methods=['POST'])
def predict_excel():
    """
    Predict churn from uploaded Excel file.
    Expects columns: customer_id, tenure, monthly_charges, total_charges,
    contract_type, payment_method, internet_service, tech_support_calls,
    has_premium_support, price_shock_percent, usage_spike_3month, data_usage_gb, overage_charges
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        return jsonify({'error': 'Only Excel (.xlsx, .xls) or CSV files allowed'}), 400
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Validate required columns
        required_cols = ['customer_id', 'tenure', 'monthly_charges', 'total_charges',
                        'contract_type', 'payment_method', 'internet_service']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        
        # Fill optional columns with defaults if missing
        defaults = {
            'tech_support_calls': 0,
            'has_premium_support': 0,
            'price_shock_percent': 0.0,
            'usage_spike_3month': 0,
            'data_usage_gb': 50.0,
            'overage_charges': 0.0
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Ensure correct data types
        df['tenure'] = df['tenure'].astype(int)
        df['tech_support_calls'] = df['tech_support_calls'].astype(int)
        df['has_premium_support'] = df['has_premium_support'].astype(int)
        df['usage_spike_3month'] = df['usage_spike_3month'].astype(int)
        
        # Get predictions
        predictions = predictor.predict_and_explain(df)
        
        # Build results DataFrame
        results = []
        for pred in predictions:
            results.append({
                'customer_id': pred.customer_id,
                'churn_probability': round(pred.churn_probability, 4),
                'churn_probability_pct': f"{pred.churn_probability:.1%}",
                'risk_level': pred.risk_level,
                'churn_prediction': 1 if pred.churn_probability >= 0.5 else 0,
                'reason_1': pred.top_reasons[0] if len(pred.top_reasons) > 0 else '',
                'reason_2': pred.top_reasons[1] if len(pred.top_reasons) > 1 else '',
                'reason_3': pred.top_reasons[2] if len(pred.top_reasons) > 2 else ''
            })
        
        results_df = pd.DataFrame(results)
        
        # Save to temp file for download
        output_path = 'predictions_output.csv'
        results_df.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'total': len(results),
            'high_risk': len([r for r in results if r['risk_level'] == 'High']),
            'medium_risk': len([r for r in results if r['risk_level'] == 'Medium']),
            'low_risk': len([r for r in results if r['risk_level'] == 'Low']),
            'preview': results[:10],
            'download_url': '/api/download/predictions'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/predictions')
def download_predictions():
    """Download predictions as CSV."""
    try:
        return send_file('predictions_output.csv', 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name='churn_predictions.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/template')
def download_template():
    """Download Excel template for bulk upload."""
    try:
        # Create template DataFrame
        template_data = {
            'customer_id': ['CUST_001', 'CUST_002'],
            'tenure': [12, 24],
            'monthly_charges': [65.00, 80.50],
            'total_charges': [780.00, 1932.00],
            'contract_type': ['Month-to-Month', 'One Year'],
            'payment_method': ['Electronic Check', 'Credit Card'],
            'internet_service': ['Fiber Optic', 'DSL'],
            'tech_support_calls': [2, 0],
            'has_premium_support': [0, 1],
            'price_shock_percent': [15.5, 0.0],
            'usage_spike_3month': [1, 0],
            'data_usage_gb': [75.5, 30.0],
            'overage_charges': [25.00, 0.00]
        }
        
        template_df = pd.DataFrame(template_data)
        template_path = 'upload_template.xlsx'
        template_df.to_excel(template_path, index=False)
        
        return send_file(template_path,
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name='churn_upload_template.xlsx')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Improvement #6: Data Drift Detection API ---

@app.route('/api/drift', methods=['POST'])
@require_api_key
def check_drift():
    """
    Check for data drift by comparing uploaded data to training distribution.
    Accepts JSON array of customer features.
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Expected array of customers'}), 400
    
    try:
        rows = []
        for item in data:
            rows.append({
                'tenure': int(item.get('tenure', 12)),
                'monthly_charges': float(item.get('monthly_charges', 65.0)),
                'total_charges': float(item.get('total_charges', 780.0)),
                'price_shock_percent': float(item.get('price_shock_percent', 0)),
                'tech_support_calls': int(item.get('tech_support_calls', 0)),
                'data_usage_gb': float(item.get('data_usage_gb', 50)),
                'overage_charges': float(item.get('overage_charges', 0)),
                'contract_type': item.get('contract_type', 'Month-to-Month'),
                'churn': int(item.get('churn', 0))
            })
        
        new_data = pd.DataFrame(rows)
        drift_result = predictor.detect_drift(new_data)
        return jsonify(drift_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Improvement #10: Model Performance Monitoring API ---

@app.route('/api/monitoring/performance')
@require_api_key
def get_performance():
    """Get model performance history and degradation check."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    history = predictor.get_performance_history()
    degradation = predictor.check_performance_degradation()
    
    return jsonify({
        'current_status': degradation,
        'history': history[-10:]  # Last 10 measurements
    })


@app.route('/api/monitoring/drift-history')
@require_api_key
def get_drift_history():
    """Get historical drift check results."""
    Drift = Query()
    records = predictor.monitoring_db.search(Drift.type == 'drift_check')
    return jsonify(records[-20:])


@app.route('/api/monitoring/alerts')
@require_api_key
def get_alert_history():
    """Get historical alert records."""
    Alert = Query()
    records = predictor.monitoring_db.search(Alert.type == 'alert')
    return jsonify(records[-20:])


# --- Improvement #5: Alert Trigger API ---

@app.route('/api/alerts/trigger', methods=['POST'])
@require_api_key
def trigger_alert():
    """
    Manually trigger an alert check.
    Accepts JSON with optional recipients and smtp_config.
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json() or {}
    
    # Get current high-risk count from DB
    Customer = Query()
    high_risk = db.search(Customer.risk_level == 'High')
    
    threshold = data.get('threshold', 10)
    recipients = data.get('recipients')
    smtp_config = data.get('smtp_config')
    
    if smtp_config:
        smtp_config = {
            'host': smtp_config.get('host', os.environ.get('SMTP_HOST', '')),
            'port': int(smtp_config.get('port', os.environ.get('SMTP_PORT', 587))),
            'sender': smtp_config.get('sender', os.environ.get('SMTP_SENDER', '')),
            'password': smtp_config.get('password', os.environ.get('SMTP_PASSWORD', ''))
        }
    
    predictor.check_and_alert(
        predictions=[],  # Will use DB count
        threshold=threshold,
        recipients=recipients,
        smtp_config=smtp_config
    )
    
    return jsonify({
        'high_risk_count': len(high_risk),
        'threshold': threshold,
        'alert_triggered': len(high_risk) >= threshold
    })


# --- Improvement #3: ROI Summary API ---

@app.route('/api/roi-summary')
@require_api_key
def get_roi_summary():
    """Get aggregate ROI analysis across all high-risk customers."""
    Customer = Query()
    high_risk = db.search(Customer.risk_level == 'High')
    
    if not high_risk:
        return jsonify({'total_customers': 0})
    
    total_loss = sum(r.get('expected_revenue_loss', 0) for r in high_risk)
    total_cost = sum(r.get('retention_cost', 0) for r in high_risk)
    rois = [r.get('retention_roi', 0) for r in high_risk if r.get('retention_roi', 0) > 0]
    avg_roi = np.mean(rois) if rois else 0
    
    return jsonify({
        'total_high_risk': len(high_risk),
        'total_expected_revenue_loss': round(total_loss, 2),
        'total_retention_investment': round(total_cost, 2),
        'net_savings_potential': round(total_loss - total_cost, 2),
        'average_retention_roi': round(avg_roi, 2),
        'positive_roi_customers': len(rois)
    })


# --- Improvement #4: Risk Timeline API ---

@app.route('/api/customer/<customer_id>/timeline')
def get_customer_timeline(customer_id):
    """Get risk timeline for a specific customer."""
    Customer = Query()
    record = db.search(Customer.customer_id == customer_id)
    
    if not record:
        return jsonify({'error': 'Customer not found'}), 404
    
    return jsonify({
        'customer_id': customer_id,
        'risk_history': record[0].get('risk_history', []),
        'current_probability': record[0].get('churn_probability', 0),
        'current_risk_level': record[0].get('risk_level', 'Unknown')
    })


# --- Improvement #9: Batch Run API ---

@app.route('/api/batch/run', methods=['POST'])
@require_api_key
def run_batch():
    """
    Trigger a batch prediction run.
    Optionally accepts data_path in JSON body.
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json() or {}
    data_path = data.get('data_path', 'telecom_data.csv')
    
    try:
        from train_and_explain import nightly_batch_predict
        predictions = nightly_batch_predict(data_path=data_path)
        
        if predictions is None:
            return jsonify({'error': 'Batch failed - no data file found'}), 500
        
        return jsonify({
            'success': True,
            'total': len(predictions),
            'high_risk': len([p for p in predictions if p.risk_level == 'High']),
            'medium_risk': len([p for p in predictions if p.risk_level == 'Medium']),
            'low_risk': len([p for p in predictions if p.risk_level == 'Low'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
