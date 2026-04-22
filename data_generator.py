"""
Synthetic Telecom Data Generator
Generates realistic telecom customer data with churn indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TelecomDataGenerator:
    """Generates synthetic telecom customer data for churn prediction."""
    
    def __init__(self, n_samples: int = 5000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _generate_base_features(self) -> pd.DataFrame:
        """Generate base customer features."""
        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(self.n_samples)],
            'tenure': np.random.exponential(24, self.n_samples).clip(1, 72).astype(int),
            'monthly_charges': np.random.normal(65, 30, self.n_samples).clip(20, 120).round(2),
            'contract_type': np.random.choice(
                ['Month-to-Month', 'One Year', 'Two Year'], 
                self.n_samples, 
                p=[0.55, 0.25, 0.20]
            ),
            'payment_method': np.random.choice(
                ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
                self.n_samples,
                p=[0.35, 0.20, 0.25, 0.20]
            ),
            'internet_service': np.random.choice(
                ['DSL', 'Fiber Optic', 'No'],
                self.n_samples,
                p=[0.35, 0.45, 0.20]
            ),
            'tech_support_calls': np.random.poisson(1.5, self.n_samples),
            'has_premium_support': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
        }
        return pd.DataFrame(data)
    
    def _calculate_price_shock(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate percentage change in last month's bill vs previous 6-month average.
        This is often the #1 predictor of churn in telecom.
        """
        # Generate 6 months of historical billing data
        base_charges = df['monthly_charges'].values
        
        # Historical months (with some variation)
        hist_bills = np.array([
            base_charges * np.random.normal(1.0, 0.05, self.n_samples)
            for _ in range(6)
        ])
        
        avg_6month = hist_bills.mean(axis=0)
        
        # Last month bill (sometimes with intentional spikes)
        spike_prob = 0.15
        spike_mask = np.random.random(self.n_samples) < spike_prob
        last_month = base_charges.copy()
        last_month[spike_mask] *= np.random.uniform(1.3, 2.0, spike_mask.sum())
        
        price_shock = ((last_month - avg_6month) / avg_6month * 100).round(2)
        return pd.Series(price_shock, index=df.index)
    
    def _generate_usage_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate usage spike indicators."""
        df['usage_spike_3month'] = np.random.choice([0, 1], self.n_samples, p=[0.75, 0.25])
        df['data_usage_gb'] = np.random.exponential(50, self.n_samples).clip(5, 300).round(1)
        df['overage_charges'] = np.where(
            df['data_usage_gb'] > 100,
            (df['data_usage_gb'] - 100) * np.random.uniform(5, 15, self.n_samples),
            0
        ).round(2)
        return df
    
    def _generate_churn_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate churn labels based on realistic business logic.
        Higher churn probability for:
        - Low tenure customers
        - Month-to-month contracts
        - High price shocks
        - High tech support calls (frustration indicator)
        - Usage spikes with overage charges
        """
        churn_prob = np.zeros(self.n_samples)
        
        # Tenure effect (inverse - lower tenure = higher churn)
        churn_prob += np.clip((72 - df['tenure']) / 72 * 0.25, 0, 0.25)
        
        # Contract type effect
        churn_prob += (df['contract_type'] == 'Month-to-Month') * 0.20
        churn_prob -= (df['contract_type'] == 'Two Year') * 0.15
        
        # Price shock effect
        churn_prob += np.clip(df['price_shock_percent'] / 100, 0, 0.25)
        
        # Tech support frustration
        churn_prob += np.clip(df['tech_support_calls'] / 10, 0, 0.15)
        
        # Usage spike with overage
        churn_prob += (df['usage_spike_3month'] & (df['overage_charges'] > 0)) * 0.10
        
        # Payment method (electronic check correlates with higher churn)
        churn_prob += (df['payment_method'] == 'Electronic Check') * 0.08
        
        # Premium support reduces churn
        churn_prob -= df['has_premium_support'] * 0.10
        
        # Clip probability between 0.05 and 0.95
        churn_prob = np.clip(churn_prob, 0.05, 0.95)
        
        return pd.Series(np.random.random(self.n_samples) < churn_prob, dtype=int)
    
    def generate(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate complete synthetic telecom dataset.
        
        Args:
            output_path: Optional path to save CSV
            
        Returns:
            DataFrame with all features and churn label
        """
        df = self._generate_base_features()
        df['price_shock_percent'] = self._calculate_price_shock(df)
        df = self._generate_usage_patterns(df)
        df['churn'] = self._generate_churn_labels(df)
        
        # Calculate total charges (tenure * monthly with some variation)
        df['total_charges'] = (df['tenure'] * df['monthly_charges'] * 
                               np.random.uniform(0.9, 1.1, self.n_samples)).round(2)
        
        # Reorder columns logically
        column_order = [
            'customer_id', 'churn', 'tenure', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'internet_service',
            'price_shock_percent', 'tech_support_calls', 'has_premium_support',
            'usage_spike_3month', 'data_usage_gb', 'overage_charges'
        ]
        df = df[column_order]
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Dataset saved to {output_path}")
            print(f"Shape: {df.shape}")
            print(f"Churn rate: {df['churn'].mean():.1%}")
        
        return df


def generate_dataset(n_samples: int = 5000, output_path: str = 'telecom_data.csv') -> pd.DataFrame:
    """Convenience function to generate dataset."""
    generator = TelecomDataGenerator(n_samples=n_samples)
    return generator.generate(output_path)


if __name__ == '__main__':
    # Generate sample dataset
    df = generate_dataset(n_samples=5000, output_path='telecom_data.csv')
    print("\nSample data:")
    print(df.head(10))
    print(f"\nClass distribution:")
    print(df['churn'].value_counts(normalize=True))
