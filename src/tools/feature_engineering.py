import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging

class FourierTransformTool(BaseTool):
    name = "fourier_analysis"
    description = "Analyzes trading frequency patterns using FFT"
    
    def _run(self, transactions: List[Dict]) -> Dict[str, Any]:
        if not transactions or len(transactions) < 10:
            return {
                'dominant_frequencies': [],
                'pattern_type': 'insufficient_data',
                'periodicity_days': None,
                'frequency_strength': 0.0
            }
        
        try:
            time_series = self._create_time_series(transactions)
            
            if len(time_series) < 10:
                return {
                    'dominant_frequencies': [],
                    'pattern_type': 'sparse_trading',
                    'periodicity_days': None,
                    'frequency_strength': 0.0
                }
            
            fft_values = fft(time_series)
            frequencies = fftfreq(len(time_series), d=1.0)
            
            magnitudes = np.abs(fft_values)
            magnitudes[0] = 0
            
            dominant_indices = np.argsort(magnitudes)[-3:][::-1]
            dominant_frequencies = frequencies[dominant_indices].tolist()
            
            pattern_type = self._interpret_pattern(dominant_frequencies[0] if dominant_frequencies else 0)
            
            periodicity_days = None
            if dominant_frequencies and dominant_frequencies[0] > 0:
                periodicity_days = 1 / dominant_frequencies[0]
            
            frequency_strength = float(magnitudes[dominant_indices[0]] / np.sum(magnitudes)) if dominant_indices.size > 0 else 0.0
            
            return {
                'dominant_frequencies': dominant_frequencies,
                'pattern_type': pattern_type,
                'periodicity_days': periodicity_days,
                'frequency_strength': frequency_strength,
                'trading_regularity': self._calculate_regularity(time_series)
            }
            
        except Exception as e:
            logging.error(f"FFT analysis error: {e}")
            return {
                'dominant_frequencies': [],
                'pattern_type': 'error',
                'periodicity_days': None,
                'frequency_strength': 0.0
            }
    
    def _create_time_series(self, transactions: List[Dict]) -> np.ndarray:
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        
        time_series = []
        for date in date_range:
            count = daily_counts.get(date.date(), 0)
            time_series.append(count)
        
        return np.array(time_series)
    
    def _interpret_pattern(self, frequency: float) -> str:
        if frequency == 0 or frequency is None:
            return 'aperiodic'
        
        period = 1 / abs(frequency)
        
        if 0.9 <= period <= 1.1:
            return 'daily'
        elif 6.5 <= period <= 7.5:
            return 'weekly'
        elif 13 <= period <= 15:
            return 'biweekly'
        elif 28 <= period <= 32:
            return 'monthly'
        else:
            return f'custom_{int(period)}_days'
    
    def _calculate_regularity(self, time_series: np.ndarray) -> float:
        if len(time_series) < 2:
            return 0.0
        
        non_zero = time_series[time_series > 0]
        if len(non_zero) < 2:
            return 0.0
        
        cv = np.std(non_zero) / np.mean(non_zero) if np.mean(non_zero) > 0 else 1.0
        
        regularity = max(0, 1 - cv)
        return float(regularity)

class AutocorrelationTool(BaseTool):
    name = "autocorrelation_analysis"
    description = "Calculate autocorrelation for trading behavior patterns"
    
    def _run(self, transactions: List[Dict]) -> Dict[str, Any]:
        if not transactions or len(transactions) < 30:
            return {
                'correlation_lag_1': 0.0,
                'correlation_lag_7': 0.0,
                'correlation_lag_30': 0.0,
                'identified_period': None,
                'pattern_strength': 0.0
            }
        
        try:
            time_series = self._create_value_series(transactions)
            
            if len(time_series) < 30:
                return {
                    'correlation_lag_1': 0.0,
                    'correlation_lag_7': 0.0,
                    'correlation_lag_30': 0.0,
                    'identified_period': None,
                    'pattern_strength': 0.0
                }
            
            correlations = {}
            for lag in [1, 7, 30]:
                if lag < len(time_series):
                    correlation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                    correlations[f'correlation_lag_{lag}'] = float(correlation) if not np.isnan(correlation) else 0.0
                else:
                    correlations[f'correlation_lag_{lag}'] = 0.0
            
            all_correlations = []
            for lag in range(1, min(31, len(time_series))):
                corr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                if not np.isnan(corr):
                    all_correlations.append((lag, corr))
            
            identified_period = None
            pattern_strength = 0.0
            
            if all_correlations:
                all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                if abs(all_correlations[0][1]) > 0.3:
                    identified_period = all_correlations[0][0]
                    pattern_strength = abs(all_correlations[0][1])
            
            return {
                **correlations,
                'identified_period': identified_period,
                'pattern_strength': float(pattern_strength),
                'trading_consistency': self._calculate_consistency(time_series)
            }
            
        except Exception as e:
            logging.error(f"Autocorrelation error: {e}")
            return {
                'correlation_lag_1': 0.0,
                'correlation_lag_7': 0.0,
                'correlation_lag_30': 0.0,
                'identified_period': None,
                'pattern_strength': 0.0
            }
    
    def _create_value_series(self, transactions: List[Dict]) -> np.ndarray:
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        daily_values = df.groupby(df['timestamp'].dt.date)['value_usd'].sum()
        
        time_series = []
        for date in date_range:
            value = daily_values.get(date.date(), 0)
            time_series.append(value)
        
        return np.array(time_series)
    
    def _calculate_consistency(self, time_series: np.ndarray) -> float:
        active_days = np.sum(time_series > 0)
        total_days = len(time_series)
        
        if total_days == 0:
            return 0.0
        
        return float(active_days / total_days)

class StatisticalFeatureTool(BaseTool):
    name = "statistical_features"
    description = "Calculate statistical features from transaction data"
    
    def _run(self, transactions: List[Dict]) -> Dict[str, Any]:
        if not transactions:
            return self._get_default_features()
        
        try:
            df = pd.DataFrame(transactions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            amounts = df['value_usd'].values
            
            features = {
                'mean_transaction_amount': float(np.mean(amounts)),
                'std_transaction_amount': float(np.std(amounts)),
                'median_transaction_amount': float(np.median(amounts)),
                'skewness_transaction_amount': float(stats.skew(amounts)),
                'kurtosis_transaction_amount': float(stats.kurtosis(amounts)),
                
                'transaction_count': len(transactions),
                'transaction_count_24h': self._count_recent_transactions(df, hours=24),
                'transaction_count_7d': self._count_recent_transactions(df, days=7),
                'transaction_count_30d': self._count_recent_transactions(df, days=30),
                
                'whale_transaction_ratio': self._calculate_whale_ratio(amounts),
                'micro_transaction_ratio': self._calculate_micro_ratio(amounts),
                
                'hour_distribution': self._get_hour_distribution(df),
                'weekday_distribution': self._get_weekday_distribution(df),
                
                'avg_time_between_transactions': self._calculate_avg_time_between(df),
                'transaction_velocity': self._calculate_velocity(df),
                
                'unique_protocols': len(df['protocol'].unique()) if 'protocol' in df else 0,
                'most_used_protocol': df['protocol'].mode()[0] if 'protocol' in df and not df['protocol'].empty else None,
                
                'gas_efficiency': self._calculate_gas_efficiency(df),
                'avg_gas_per_transaction': float(df['gas_paid'].mean()) if 'gas_paid' in df else 0.0
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Statistical feature extraction error: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, Any]:
        return {
            'mean_transaction_amount': 0.0,
            'std_transaction_amount': 0.0,
            'median_transaction_amount': 0.0,
            'skewness_transaction_amount': 0.0,
            'kurtosis_transaction_amount': 0.0,
            'transaction_count': 0,
            'transaction_count_24h': 0,
            'transaction_count_7d': 0,
            'transaction_count_30d': 0,
            'whale_transaction_ratio': 0.0,
            'micro_transaction_ratio': 0.0,
            'hour_distribution': {},
            'weekday_distribution': {},
            'avg_time_between_transactions': 0.0,
            'transaction_velocity': 0.0,
            'unique_protocols': 0,
            'most_used_protocol': None,
            'gas_efficiency': 0.0,
            'avg_gas_per_transaction': 0.0
        }
    
    def _count_recent_transactions(self, df: pd.DataFrame, hours: int = 0, days: int = 0) -> int:
        cutoff = datetime.now() - timedelta(hours=hours, days=days)
        return len(df[df['timestamp'] > cutoff])
    
    def _calculate_whale_ratio(self, amounts: np.ndarray) -> float:
        whale_threshold = 100000
        whale_count = np.sum(amounts >= whale_threshold)
        return float(whale_count / len(amounts)) if len(amounts) > 0 else 0.0
    
    def _calculate_micro_ratio(self, amounts: np.ndarray) -> float:
        micro_threshold = 100
        micro_count = np.sum(amounts <= micro_threshold)
        return float(micro_count / len(amounts)) if len(amounts) > 0 else 0.0
    
    def _get_hour_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        hour_counts = df['timestamp'].dt.hour.value_counts().to_dict()
        return {int(k): int(v) for k, v in hour_counts.items()}
    
    def _get_weekday_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        weekday_counts = df['timestamp'].dt.dayofweek.value_counts().to_dict()
        return {int(k): int(v) for k, v in weekday_counts.items()}
    
    def _calculate_avg_time_between(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        
        if time_diffs.empty:
            return 0.0
        
        avg_diff = time_diffs.mean()
        return float(avg_diff.total_seconds() / 3600)
    
    def _calculate_velocity(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
        
        if time_span == 0:
            return 0.0
        
        return float(len(df) / max(time_span, 1))
    
    def _calculate_gas_efficiency(self, df: pd.DataFrame) -> float:
        if 'gas_paid' not in df or 'value_usd' not in df:
            return 0.0
        
        total_value = df['value_usd'].sum()
        total_gas = df['gas_paid'].sum()
        
        if total_value == 0:
            return 0.0
        
        return float(1 - (total_gas / total_value)) if total_gas < total_value else 0.0

class PortfolioFeatureTool(BaseTool):
    name = "portfolio_features"
    description = "Extract portfolio-level features from wallet data"
    
    def _run(self, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            balances = wallet_data.get('balances', {})
            
            if not balances:
                return self._get_default_portfolio_features()
            
            allocations = [b.get('allocation_percentage', 0) for b in balances.values()]
            values = [b.get('value_usd', 0) for b in balances.values()]
            
            hhi = sum([(a/100)**2 for a in allocations]) if allocations else 0
            
            features = {
                'portfolio_diversity': float(1 - hhi),
                'top_token_concentration': max(allocations) if allocations else 0.0,
                'portfolio_value': sum(values),
                'token_count': len(balances),
                'average_position_size': np.mean(values) if values else 0.0,
                'largest_position': max(values) if values else 0.0,
                'smallest_position': min(values) if values else 0.0,
                'position_ratio': max(values) / min(values) if values and min(values) > 0 else 0.0,
                'stablecoin_percentage': self._calculate_stablecoin_percentage(balances),
                'defi_token_percentage': self._calculate_defi_percentage(balances)
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Portfolio feature extraction error: {e}")
            return self._get_default_portfolio_features()
    
    def _get_default_portfolio_features(self) -> Dict[str, Any]:
        return {
            'portfolio_diversity': 0.0,
            'top_token_concentration': 0.0,
            'portfolio_value': 0.0,
            'token_count': 0,
            'average_position_size': 0.0,
            'largest_position': 0.0,
            'smallest_position': 0.0,
            'position_ratio': 0.0,
            'stablecoin_percentage': 0.0,
            'defi_token_percentage': 0.0
        }
    
    def _calculate_stablecoin_percentage(self, balances: Dict) -> float:
        stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP']
        
        stable_value = sum(
            b.get('value_usd', 0) 
            for token, b in balances.items() 
            if token.upper() in stablecoins
        )
        
        total_value = sum(b.get('value_usd', 0) for b in balances.values())
        
        if total_value == 0:
            return 0.0
        
        return float(stable_value / total_value * 100)
    
    def _calculate_defi_percentage(self, balances: Dict) -> float:
        defi_tokens = ['UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI', 'CRV', 'BAL']
        
        defi_value = sum(
            b.get('value_usd', 0) 
            for token, b in balances.items() 
            if token.upper() in defi_tokens
        )
        
        total_value = sum(b.get('value_usd', 0) for b in balances.values())
        
        if total_value == 0:
            return 0.0
        
        return float(defi_value / total_value * 100)