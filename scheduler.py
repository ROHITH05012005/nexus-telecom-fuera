"""
Nightly Batch Scheduler for Telecom Churn Prediction.
Uses APScheduler to run predictions automatically.

Usage:
    python scheduler.py

Or add to crontab (Linux/Mac):
    0 2 * * * cd /path/to/project && python -c "from scheduler import run_nightly; run_nightly()"
"""

import os
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler


def run_nightly():
    """Run nightly batch prediction."""
    from train_and_explain import nightly_batch_predict
    
    # SMTP config from environment variables
    smtp_config = None
    if os.environ.get('SMTP_HOST'):
        smtp_config = {
            'host': os.environ.get('SMTP_HOST'),
            'port': int(os.environ.get('SMTP_PORT', 587)),
            'sender': os.environ.get('SMTP_SENDER', ''),
            'password': os.environ.get('SMTP_PASSWORD', '')
        }
    
    recipients = None
    if os.environ.get('ALERT_RECIPIENTS'):
        recipients = os.environ.get('ALERT_RECIPIENTS').split(',')
    
    nightly_batch_predict(
        data_path='telecom_data.csv',
        db_path='at_risk_customers.json',
        alert_threshold=10,
        recipients=recipients,
        smtp_config=smtp_config
    )


if __name__ == '__main__':
    scheduler = BlockingScheduler()
    
    # Run nightly at 2:00 AM
    scheduler.add_job(run_nightly, 'cron', hour=2, minute=0)
    
    # Also run immediately on startup for testing
    print("Running initial batch...")
    run_nightly()
    
    print("\nScheduler started. Nightly batch will run at 2:00 AM daily.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
