"""
Dynamic Time Warping analysis for transaction sequences
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from src.database import WalletRepository
from src.config import get_config

class DTWAnalyzer:
    """Dynamic Time Warping analysis for transaction sequences"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize repository (handles database connection internally)
        self.wallet_repo = WalletRepository()
    
    def analyze_transaction_sequence(self, wallet_address: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction sequence using DTW"""
        try:
            if not transactions or len(transactions) < 2:
                return self._get_default_dtw_result(wallet_address)
            
            # Extract transaction features
            sequence_features = self._extract_sequence_features(transactions)
            
            # Calculate DTW distance matrix
            dtw_matrix = self._calculate_dtw_matrix(sequence_features)
            
            # Analyze patterns
            pattern_analysis = self._analyze_patterns(sequence_features, dtw_matrix)
            
            # Calculate sequence similarity
            similarity_metrics = self._calculate_similarity_metrics(sequence_features)
            
            return {
                'sequence_length': len(transactions),
                'pattern_type': pattern_analysis['pattern_type'],
                'pattern_strength': pattern_analysis['pattern_strength'],
                'dtw_distance': float(pattern_analysis['avg_dtw_distance']),
                'similarity_score': float(similarity_metrics['similarity_score']),
                'periodicity_days': pattern_analysis.get('periodicity_days'),
                'trend_direction': pattern_analysis['trend_direction'],
                'volatility_score': float(similarity_metrics['volatility_score']),
                'sequence_characteristics': pattern_analysis['characteristics'],
                'analysis_timestamp': datetime.now().isoformat(),
                'wallet_address': wallet_address,
                'transaction_count': len(transactions)
            }
            
        except Exception as e:
            self.logger.error(f"DTW analysis failed: {e}")
            return self._get_default_dtw_result(wallet_address)
    
    def _extract_sequence_features(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from transaction sequence"""
        try:
            features = []
            
            for tx in transactions:
                # Extract transaction features
                amount = float(tx.get('amount', tx.get('value_usd', 0)))
                timestamp = self._parse_timestamp(tx.get('timestamp', ''))
                
                # Normalize amount (log scale for large variations)
                normalized_amount = np.log10(max(amount, 1))
                
                # Time-based features
                hour = timestamp.hour if timestamp else 12
                day_of_week = timestamp.weekday() if timestamp else 3
                
                # Protocol features
                protocol = tx.get('protocol', 'unknown')
                protocol_encoded = hash(protocol) % 100 / 100.0  # Simple encoding
                
                # Transaction type features
                tx_type = tx.get('type', 'unknown')
                type_encoded = hash(tx_type) % 100 / 100.0
                
                # Combine features
                feature_vector = [
                    normalized_amount,
                    hour / 24.0,  # Normalize hour
                    day_of_week / 7.0,  # Normalize day
                    protocol_encoded,
                    type_encoded
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([[0, 0, 0, 0, 0]])
    
    def _calculate_dtw_matrix(self, sequence_features: np.ndarray) -> np.ndarray:
        """Calculate DTW distance matrix"""
        try:
            n = len(sequence_features)
            dtw_matrix = np.full((n, n), np.inf)
            
            # Initialize diagonal
            for i in range(n):
                dtw_matrix[i, i] = 0
            
            # Calculate DTW distances
            for i in range(n):
                for j in range(i + 1, n):
                    distance = self._dtw_distance(sequence_features[i:j+1])
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance  # Symmetric
            
            return dtw_matrix
            
        except Exception as e:
            self.logger.error(f"DTW matrix calculation failed: {e}")
            return np.array([[0]])
    
    def _dtw_distance(self, sequence: np.ndarray) -> float:
        """Calculate DTW distance for a sequence"""
        try:
            if len(sequence) < 2:
                return 0.0
            
            # Simple DTW implementation
            n = len(sequence)
            dtw_matrix = np.full((n, n), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(n):
                for j in range(n):
                    if i == 0 and j == 0:
                        continue
                    
                    # Current cost
                    cost = euclidean(sequence[i], sequence[j])
                    
                    # Previous minimum
                    prev_min = np.inf
                    if i > 0:
                        prev_min = min(prev_min, dtw_matrix[i-1, j])
                    if j > 0:
                        prev_min = min(prev_min, dtw_matrix[i, j-1])
                    if i > 0 and j > 0:
                        prev_min = min(prev_min, dtw_matrix[i-1, j-1])
                    
                    dtw_matrix[i, j] = cost + prev_min
            
            return float(dtw_matrix[n-1, n-1])
            
        except Exception as e:
            self.logger.error(f"DTW distance calculation failed: {e}")
            return 0.0
    
    def _analyze_patterns(self, sequence_features: np.ndarray, dtw_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze patterns in the sequence"""
        try:
            # Calculate average DTW distance
            non_inf_distances = dtw_matrix[dtw_matrix != np.inf]
            avg_dtw_distance = np.mean(non_inf_distances) if len(non_inf_distances) > 0 else 0
            
            # Analyze amount patterns
            amounts = sequence_features[:, 0]  # First feature is amount
            amount_std = np.std(amounts)
            amount_mean = np.mean(amounts)
            
            # Analyze time patterns
            hours = sequence_features[:, 1] * 24  # Convert back to hours
            days = sequence_features[:, 2] * 7    # Convert back to days
            
            # Pattern classification
            if amount_std < 0.5:
                pattern_type = 'consistent'
                pattern_strength = 0.8
            elif amount_std < 1.0:
                pattern_type = 'moderate_variation'
                pattern_strength = 0.6
            else:
                pattern_type = 'high_variation'
                pattern_strength = 0.4
            
            # Trend analysis
            if len(amounts) > 1:
                trend_coef = np.polyfit(range(len(amounts)), amounts, 1)[0]
                if trend_coef > 0.1:
                    trend_direction = 'increasing'
                elif trend_coef < -0.1:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'unknown'
            
            # Periodicity detection
            periodicity_days = self._detect_periodicity(days)
            
            # Characteristics
            characteristics = []
            if amount_std < 0.3:
                characteristics.append('consistent_amounts')
            if np.std(hours) < 4:
                characteristics.append('regular_timing')
            if periodicity_days:
                characteristics.append(f'periodic_{periodicity_days}d')
            if trend_direction != 'stable':
                characteristics.append(f'trend_{trend_direction}')
            
            return {
                'pattern_type': pattern_type,
                'pattern_strength': float(pattern_strength),
                'avg_dtw_distance': float(avg_dtw_distance),
                'trend_direction': trend_direction,
                'periodicity_days': periodicity_days,
                'amount_volatility': float(amount_std),
                'characteristics': characteristics
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {
                'pattern_type': 'unknown',
                'pattern_strength': 0.0,
                'avg_dtw_distance': 0.0,
                'trend_direction': 'unknown',
                'characteristics': []
            }
    
    def _detect_periodicity(self, days: np.ndarray) -> Optional[int]:
        """Detect periodicity in transaction days"""
        try:
            if len(days) < 4:
                return None
            
            # Calculate day differences
            day_diffs = np.diff(days)
            
            # Look for common patterns
            unique_diffs, counts = np.unique(day_diffs, return_counts=True)
            
            # Find most common difference
            if len(unique_diffs) > 0:
                most_common_diff = unique_diffs[np.argmax(counts)]
                most_common_count = np.max(counts)
                
                # Check if it's a significant pattern
                if most_common_count >= len(day_diffs) * 0.3:  # 30% threshold
                    return int(most_common_diff)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Periodicity detection failed: {e}")
            return None
    
    def _calculate_similarity_metrics(self, sequence_features: np.ndarray) -> Dict[str, float]:
        """Calculate similarity metrics for the sequence"""
        try:
            # Calculate feature correlations
            feature_correlations = []
            for i in range(sequence_features.shape[1]):
                for j in range(i + 1, sequence_features.shape[1]):
                    corr = np.corrcoef(sequence_features[:, i], sequence_features[:, j])[0, 1]
                    if not np.isnan(corr):
                        feature_correlations.append(corr)
            
            # Overall similarity score
            similarity_score = np.mean(feature_correlations) if feature_correlations else 0.0
            
            # Volatility score
            amounts = sequence_features[:, 0]
            volatility_score = np.std(amounts) / (np.mean(amounts) + 1e-6)
            
            return {
                'similarity_score': float(similarity_score),
                'volatility_score': float(volatility_score),
                'feature_correlation_count': len(feature_correlations)
            }
            
        except Exception as e:
            self.logger.error(f"Similarity metrics calculation failed: {e}")
            return {
                'similarity_score': 0.0,
                'volatility_score': 0.0,
                'feature_correlation_count': 0
            }
    
    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """Parse timestamp to datetime"""
        try:
            if not timestamp:
                return None
            
            # If already a datetime object, return it
            if isinstance(timestamp, datetime):
                return timestamp
            
            # If it's a string, parse it
            if isinstance(timestamp, str):
                # Handle different timestamp formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except ValueError:
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Timestamp parsing failed: {e}")
            return None
    
    def _get_default_dtw_result(self, wallet_address: str) -> Dict[str, Any]:
        """Get default DTW analysis result"""
        return {
            'sequence_length': 0,
            'pattern_type': 'unknown',
            'pattern_strength': 0.0,
            'dtw_distance': 0.0,
            'similarity_score': 0.0,
            'periodicity_days': None,
            'trend_direction': 'unknown',
            'volatility_score': 0.0,
            'sequence_characteristics': [],
            'analysis_timestamp': datetime.now().isoformat(),
            'wallet_address': wallet_address,
            'transaction_count': 0,
            'error': 'Insufficient data for DTW analysis'
        }
    
    def compare_sequences(self, sequence1: List[Dict[str, Any]], 
                         sequence2: List[Dict[str, Any]]) -> float:
        """Compare two transaction sequences using DTW"""
        try:
            # Extract features for both sequences
            features1 = self._extract_sequence_features(sequence1)
            features2 = self._extract_sequence_features(sequence2)
            
            # Calculate DTW distance
            dtw_distance = self._dtw_distance_sequences(features1, features2)
            
            # Convert to similarity score
            similarity = 1.0 / (1.0 + dtw_distance)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Sequence comparison failed: {e}")
            return 0.0
    
    def _dtw_distance_sequences(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate DTW distance between two sequences"""
        try:
            n, m = len(seq1), len(seq2)
            
            # Initialize DTW matrix
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # Fill DTW matrix
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = euclidean(seq1[i-1], seq2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],    # insertion
                        dtw_matrix[i, j-1],    # deletion
                        dtw_matrix[i-1, j-1]   # match
                    )
            
            return float(dtw_matrix[n, m])
            
        except Exception as e:
            self.logger.error(f"DTW sequence distance calculation failed: {e}")
            return 0.0
    
    def close(self):
        """Close database connection"""
        # No explicit close needed here as WalletRepository handles its own connection
        pass
