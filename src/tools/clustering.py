"""
Production wallet clustering tool using database backend
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from src.config import get_config
from src.clustering import ClusterManager

class BaseClusteringTool:
    """Base class for clustering tools with common functionality"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize cluster manager
        try:
            self.cluster_manager = ClusterManager()
            self.cluster_manager_available = True
        except Exception as e:
            self.logger.warning(f"Cluster manager not available: {e}")
            self.cluster_manager_available = False
            self.cluster_manager = None

class WalletClusteringTool(BaseClusteringTool):
    name = "wallet_clustering"
    description = "Clusters wallets by behavior patterns using production database backend"
    
    def __init__(self):
        BaseClusteringTool.__init__(self)
    
    def _run(self, wallet_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster wallet based on behavior features using production database backend
        
        Args:
            wallet_features: Dictionary containing wallet behavior features
            
        Returns:
            Dictionary with clustering results
        """
        try:
            wallet_address = wallet_features.get('address', 'unknown')
            
            # Store wallet features in database
            if self.cluster_manager_available:
                self.cluster_manager.store_wallet_features(wallet_address, wallet_features)
                
                # Get clustering results from database
                clustering_result = self.cluster_manager.get_wallet_clustering(wallet_address)
                
                # Add DTW analysis if transactions are available
                transactions = wallet_features.get('transactions', [])
                if transactions:
                    dtw_result = self.cluster_manager.analyze_transaction_sequence(wallet_address, transactions)
                    clustering_result['dtw_analysis'] = dtw_result
                
                return clustering_result
            else:
                # Fallback to mock data if cluster manager not available
                return self._get_mock_clustering_result(wallet_address)
            
        except Exception as e:
            self.logger.error(f"Clustering error for wallet {wallet_features.get('address', 'unknown')}: {e}")
            return self._get_default_clustering_result(wallet_features.get('address', 'unknown'))
    
    def _get_mock_clustering_result(self, wallet_address: str) -> Dict[str, Any]:
        """Get mock clustering result when database is not available"""
        return {
            'wallet_address': wallet_address,
            'cluster_label': -1,
            'cluster_confidence': 0.0,
            'similar_wallets': [
                {
                    'address': f'0x{hash(wallet_address + "1") % 10**40:040x}',
                    'similarity_score': 0.85,
                    'shared_characteristics': ['high_frequency_trading', 'defi_protocols']
                },
                {
                    'address': f'0x{hash(wallet_address + "2") % 10**40:040x}',
                    'similarity_score': 0.78,
                    'shared_characteristics': ['similar_risk_profile', 'protocol_preferences']
                }
            ],
            'cluster_characteristics': {
                'trading_style': 'hodler',
                'risk_tolerance': 'medium',
                'protocol_preferences': ['uniswap', 'aave', 'compound'],
                'typical_holdings': ['ETH', 'USDC', 'WBTC'],
                'transaction_pattern': 'infrequent_large_trades'
            },
            'cluster_statistics': {},
            'wallet_features': {
                'trading_frequency': 0.0,
                'avg_transaction_size': 0.0,
                'risk_score': 0.5,
                'protocol_diversity': 0.0,
                'portfolio_value': 0.0,
                'transaction_count': 0
            },
            'clustering_timestamp': datetime.now().isoformat(),
            'note': 'Using mock data - database not available'
        }
    
    def _get_default_clustering_result(self, wallet_address: str) -> Dict[str, Any]:
        """Get default clustering result when analysis fails"""
        return {
            'wallet_address': wallet_address,
            'cluster_label': -1,
            'cluster_confidence': 0.0,
            'similar_wallets': [],
            'cluster_characteristics': {
                'trading_style': 'unknown',
                'risk_tolerance': 'unknown'
            },
            'distance_to_centroid': 1.0,
            'cluster_size': 1,
            'clustering_timestamp': datetime.now().isoformat(),
            'error': 'Clustering analysis failed'
        }
