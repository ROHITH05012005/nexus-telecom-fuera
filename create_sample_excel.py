"""Generate sample Excel file for testing bulk upload."""

import pandas as pd

# Create sample customer data with different risk profiles
sample_data = {
    'customer_id': [
        'TEST_001', 'TEST_002', 'TEST_003', 'TEST_004', 'TEST_005',
        'TEST_006', 'TEST_007', 'TEST_008', 'TEST_009', 'TEST_010'
    ],
    'tenure': [5, 36, 2, 48, 8, 24, 3, 60, 12, 6],
    'monthly_charges': [85.50, 45.00, 95.00, 55.00, 78.00, 62.00, 105.00, 50.00, 70.00, 88.00],
    'total_charges': [427.50, 1620.00, 190.00, 2640.00, 624.00, 1488.00, 315.00, 3000.00, 840.00, 528.00],
    'contract_type': [
        'Month-to-Month', 'Two Year', 'Month-to-Month', 'One Year', 'Month-to-Month',
        'One Year', 'Month-to-Month', 'Two Year', 'Month-to-Month', 'Month-to-Month'
    ],
    'payment_method': [
        'Electronic Check', 'Bank Transfer', 'Electronic Check', 'Credit Card',
        'Mailed Check', 'Bank Transfer', 'Electronic Check', 'Credit Card',
        'Electronic Check', 'Electronic Check'
    ],
    'internet_service': [
        'Fiber Optic', 'DSL', 'Fiber Optic', 'DSL', 'Fiber Optic',
        'DSL', 'Fiber Optic', 'DSL', 'Fiber Optic', 'Fiber Optic'
    ],
    'tech_support_calls': [3, 0, 5, 1, 2, 0, 4, 0, 1, 3],
    'has_premium_support': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    'price_shock_percent': [45.0, 0.0, 60.0, 5.0, 30.0, 0.0, 75.0, 0.0, 15.0, 50.0],
    'usage_spike_3month': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    'data_usage_gb': [120.5, 25.0, 200.0, 30.0, 95.0, 20.0, 250.0, 15.0, 60.0, 180.0],
    'overage_charges': [45.00, 0.00, 120.00, 0.00, 35.00, 0.00, 180.00, 0.00, 0.00, 95.00]
}

df = pd.DataFrame(sample_data)

# Save as Excel
output_path = 'sample_customers.xlsx'
df.to_excel(output_path, index=False, sheet_name='Customers')

print(f"Sample Excel file created: {output_path}")
print(f"Rows: {len(df)}")
print("\nPreview:")
print(df.to_string())
