"""
Similarity search for finding similar wallets
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from src.database import WalletRepository, ClusterRepository
from src.config import get_config

class SimilaritySearch:
    """Similarity search for finding similar wallets"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize repositories (they handle database connection internally)
        self.wallet_repo = WalletRepository()
        self.cluster_repo = ClusterRepository()
        
        # Similarity weights
        self.similarity_weights = {
            'feature_similarity': 0.4,
            'cluster_similarity': 0.3,
            'behavioral_similarity': 0.3
        }
    
    def find_similar_wallets(self, wallet_address: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar wallets based on multiple criteria"""
        try:
            # Get target wallet features
            target_features = self.wallet_repo.get_wallet_features(wallet_address)
            if not target_features:
                self.logger.warning(f"Wallet features not found: {wallet_address}")
                return []
            
            # Get target cluster assignment
            target_assignment = self.cluster_repo.get_cluster_assignment(wallet_address)
            
            # Get all wallets in the same cluster
            if target_assignment and target_assignment.cluster_label != -1:
                cluster_wallets = self.cluster_repo.get_wallets_by_cluster(
                    target_assignment.cluster_label, 
                    limit=limit * 2  # Get more to filter
                )
            else:
                # If no cluster assignment, get recent wallets
                cluster_wallets = self._get_recent_wallets(limit * 2)
            
            # Calculate similarities
            similar_wallets = []
            for wallet_data in cluster_wallets:
                if wallet_data['wallet_address'] == wallet_address:
                    continue
                
                similarity_score = self._calculate_similarity_score(
                    target_features, target_assignment, wallet_data
                )
                
                similar_wallets.append({
                    'address': wallet_data['wallet_address'],
                    'similarity_score': similarity_score,
                    'shared_characteristics': self._get_shared_characteristics(
                        target_features, wallet_data
                    ),
                    'cluster_confidence': wallet_data.get('cluster_confidence', 0.0),
                    'distance_to_centroid': wallet_data.get('distance_to_centroid', 0.0)
                })
            
            # Sort by similarity score and return top results
            similar_wallets.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_wallets[:limit]
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def _calculate_similarity_score(self, target_features: Any, 
                                  target_assignment: Optional[Any],
                                  wallet_data: Dict[str, Any]) -> float:
        """Calculate comprehensive similarity score"""
        try:
            # Feature similarity (Euclidean distance in feature space)
            feature_similarity = self._calculate_feature_similarity(target_features, wallet_data)
            
            # Cluster similarity
            cluster_similarity = self._calculate_cluster_similarity(target_assignment, wallet_data)
            
            # Behavioral similarity
            behavioral_similarity = self._calculate_behavioral_similarity(target_features, wallet_data)
            
            # Weighted combination
            total_similarity = (
                self.similarity_weights['feature_similarity'] * feature_similarity +
                self.similarity_weights['cluster_similarity'] * cluster_similarity +
                self.similarity_weights['behavioral_similarity'] * behavioral_similarity
            )
            
            return float(total_similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_feature_similarity(self, target_features: Any, 
                                    wallet_data: Dict[str, Any]) -> float:
        """Calculate feature-based similarity"""
        try:
            # Extract features from wallet data
            wallet_features = [
                wallet_data['trading_frequency'],
                wallet_data['avg_transaction_size'],
                wallet_data['risk_score'],
                wallet_data['protocol_diversity']
            ]
            
            target_vector = target_features.to_feature_vector()
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(target_vector - wallet_features)
            
            # Convert to similarity score (0-1)
            similarity = 1.0 / (1.0 + distance)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Feature similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_cluster_similarity(self, target_assignment: Optional[Any],
                                    wallet_data: Dict[str, Any]) -> float:
        """Calculate cluster-based similarity"""
        try:
            if not target_assignment:
                return 0.5  # Neutral score if no cluster assignment
            
            # Check if wallets are in the same cluster
            if target_assignment.cluster_label == wallet_data.get('cluster_label', -1):
                # Same cluster - high similarity
                base_similarity = 0.8
                
                # Adjust based on confidence levels
                target_confidence = target_assignment.cluster_confidence
                wallet_confidence = wallet_data.get('cluster_confidence', 0.5)
                
                confidence_factor = (target_confidence + wallet_confidence) / 2.0
                similarity = base_similarity * confidence_factor
                
            else:
                # Different clusters - lower similarity
                similarity = 0.2
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Cluster similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_behavioral_similarity(self, target_features: Any,
                                       wallet_data: Dict[str, Any]) -> float:
        """Calculate behavioral similarity"""
        try:
            similarities = []
            
            # Trading frequency similarity
            freq_diff = abs(target_features.trading_frequency - wallet_data['trading_frequency'])
            freq_similarity = 1.0 / (1.0 + freq_diff)
            similarities.append(freq_similarity)
            
            # Transaction size similarity
            size_diff = abs(target_features.avg_transaction_size - wallet_data['avg_transaction_size'])
            size_similarity = 1.0 / (1.0 + size_diff / 1000.0)  # Normalize by 1000
            similarities.append(size_similarity)
            
            # Risk tolerance similarity
            risk_diff = abs(target_features.risk_score - wallet_data['risk_score'])
            risk_similarity = 1.0 - risk_diff  # Risk scores are 0-1
            similarities.append(risk_similarity)
            
            # Protocol diversity similarity
            protocol_diff = abs(target_features.protocol_diversity - wallet_data['protocol_diversity'])
            protocol_similarity = 1.0 - protocol_diff  # Protocol diversity is 0-1
            similarities.append(protocol_similarity)
            
            # Average behavioral similarity
            behavioral_similarity = np.mean(similarities)
            
            return float(behavioral_similarity)
            
        except Exception as e:
            self.logger.error(f"Behavioral similarity calculation failed: {e}")
            return 0.0
    
    def _get_shared_characteristics(self, target_features: Any,
                                  wallet_data: Dict[str, Any]) -> List[str]:
        """Get shared behavioral characteristics"""
        try:
            characteristics = []
            
            # Trading frequency characteristics
            if abs(target_features.trading_frequency - wallet_data['trading_frequency']) < 0.1:
                if target_features.trading_frequency > 0.5:
                    characteristics.append('high_frequency_trading')
                else:
                    characteristics.append('low_frequency_trading')
            
            # Transaction size characteristics
            if abs(target_features.avg_transaction_size - wallet_data['avg_transaction_size']) < 1000:
                if target_features.avg_transaction_size > 5000:
                    characteristics.append('large_transactions')
                else:
                    characteristics.append('small_transactions')
            
            # Risk tolerance characteristics
            if abs(target_features.risk_score - wallet_data['risk_score']) < 0.2:
                if target_features.risk_score > 0.7:
                    characteristics.append('high_risk_tolerance')
                elif target_features.risk_score < 0.3:
                    characteristics.append('low_risk_tolerance')
                else:
                    characteristics.append('moderate_risk_tolerance')
            
            # Protocol diversity characteristics
            if abs(target_features.protocol_diversity - wallet_data['protocol_diversity']) < 0.2:
                if target_features.protocol_diversity > 0.5:
                    characteristics.append('diverse_protocols')
                else:
                    characteristics.append('focused_protocols')
            
            # Add cluster-based characteristics
            if wallet_data.get('cluster_label', -1) != -1:
                characteristics.append('same_cluster')
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Shared characteristics calculation failed: {e}")
            return ['similar_behavior']
    
    def _get_recent_wallets(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent wallets when no cluster assignment exists"""
        try:
            # Get recent wallet features
            recent_wallets = self.wallet_repo.get_all_wallet_features(limit=limit)
            
            # Convert to dictionary format
            wallet_data = []
            for wallet in recent_wallets:
                wallet_data.append({
                    'wallet_address': wallet.wallet_address,
                    'trading_frequency': wallet.trading_frequency,
                    'avg_transaction_size': wallet.avg_transaction_size,
                    'risk_score': wallet.risk_score,
                    'protocol_diversity': wallet.protocol_diversity,
                    'cluster_label': -1,
                    'cluster_confidence': 0.0,
                    'distance_to_centroid': 0.0
                })
            
            return wallet_data
            
        except Exception as e:
            self.logger.error(f"Failed to get recent wallets: {e}")
            return []
    
    def get_cluster_statistics(self, cluster_label: int) -> Dict[str, Any]:
        """Get statistics for a specific cluster"""
        try:
            # Get wallets in cluster
            cluster_wallets = self.cluster_repo.get_wallets_by_cluster(cluster_label, limit=1000)
            
            if not cluster_wallets:
                return {'error': 'Cluster not found'}
            
            # Calculate statistics
            trading_frequencies = [w['trading_frequency'] for w in cluster_wallets]
            transaction_sizes = [w['avg_transaction_size'] for w in cluster_wallets]
            risk_scores = [w['risk_score'] for w in cluster_wallets]
            protocol_diversities = [w['protocol_diversity'] for w in cluster_wallets]
            confidences = [w['cluster_confidence'] for w in cluster_wallets]
            
            return {
                'cluster_label': cluster_label,
                'wallet_count': len(cluster_wallets),
                'avg_trading_frequency': float(np.mean(trading_frequencies)),
                'avg_transaction_size': float(np.mean(transaction_sizes)),
                'avg_risk_score': float(np.mean(risk_scores)),
                'avg_protocol_diversity': float(np.mean(protocol_diversities)),
                'avg_confidence': float(np.mean(confidences)),
                'std_trading_frequency': float(np.std(trading_frequencies)),
                'std_transaction_size': float(np.std(transaction_sizes)),
                'std_risk_score': float(np.std(risk_scores)),
                'std_protocol_diversity': float(np.std(protocol_diversities))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster statistics: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close database connection"""
        # No explicit close needed here as repositories handle their own connections
        pass
