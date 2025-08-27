"""
Cluster manager for orchestrating clustering operations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.database import WalletRepository, ClusterRepository
from src.database.models import WalletFeatures, ClusterAssignment
from src.clustering.batch_clustering import BatchClusteringJob
from src.clustering.similarity_search import SimilaritySearch
from src.clustering.dtw_analysis import DTWAnalyzer
from src.config import get_config

class ClusterManager:
    """Manages all clustering operations and provides unified interface"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize repositories (they handle database connection internally)
        self.wallet_repo = WalletRepository()
        self.cluster_repo = ClusterRepository()
        
        self.batch_clustering = BatchClusteringJob()
        self.similarity_search = SimilaritySearch()
        self.dtw_analyzer = DTWAnalyzer()
    
    def store_wallet_features(self, wallet_address: str, wallet_data: Dict[str, Any]) -> bool:
        """Store wallet features for clustering"""
        try:
            # Extract features from wallet data
            features = self._extract_wallet_features(wallet_address, wallet_data)
            
            # Store in database
            success = self.wallet_repo.store_wallet_features(features)
            
            if success:
                self.logger.info(f"Stored features for wallet: {wallet_address}")
            else:
                self.logger.error(f"Failed to store features for wallet: {wallet_address}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store wallet features: {e}")
            return False
    
    def get_wallet_clustering(self, wallet_address: str) -> Dict[str, Any]:
        """Get comprehensive clustering information for a wallet"""
        try:
            # Get wallet features
            wallet_features = self.wallet_repo.get_wallet_features(wallet_address)
            if not wallet_features:
                return self._get_default_clustering_result(wallet_address)
            
            # Get cluster assignment
            cluster_assignment = self.cluster_repo.get_cluster_assignment(wallet_address)
            
            # Get similar wallets
            similar_wallets = self.similarity_search.find_similar_wallets(wallet_address, limit=5)
            
            # Get cluster characteristics
            cluster_characteristics = self._get_cluster_characteristics(cluster_assignment)
            
            # Get cluster statistics if available
            cluster_stats = {}
            if cluster_assignment and cluster_assignment.cluster_label != -1:
                cluster_stats = self.similarity_search.get_cluster_statistics(
                    cluster_assignment.cluster_label
                )
            
            return {
                'wallet_address': wallet_address,
                'cluster_label': cluster_assignment.cluster_label if cluster_assignment else -1,
                'cluster_confidence': cluster_assignment.cluster_confidence if cluster_assignment else 0.0,
                'distance_to_centroid': cluster_assignment.distance_to_centroid if cluster_assignment else 0.0,
                'similar_wallets': similar_wallets,
                'cluster_characteristics': cluster_characteristics,
                'cluster_statistics': cluster_stats,
                'wallet_features': {
                    'trading_frequency': wallet_features.trading_frequency,
                    'avg_transaction_size': wallet_features.avg_transaction_size,
                    'risk_score': wallet_features.risk_score,
                    'protocol_diversity': wallet_features.protocol_diversity,
                    'portfolio_value': wallet_features.portfolio_value,
                    'transaction_count': wallet_features.transaction_count
                },
                'clustering_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet clustering: {e}")
            return self._get_default_clustering_result(wallet_address)
    
    def analyze_transaction_sequence(self, wallet_address: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction sequence using DTW"""
        try:
            # Run DTW analysis
            dtw_result = self.dtw_analyzer.analyze_transaction_sequence(transactions)
            
            # Add wallet context
            dtw_result['wallet_address'] = wallet_address
            dtw_result['transaction_count'] = len(transactions)
            
            return dtw_result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze transaction sequence: {e}")
            return {
                'wallet_address': wallet_address,
                'error': 'DTW analysis failed',
                'transaction_count': len(transactions)
            }
    
    def run_batch_clustering(self) -> Dict[str, Any]:
        """Run batch clustering job"""
        try:
            self.logger.info("Starting batch clustering job")
            result = self.batch_clustering.run()
            self.logger.info(f"Batch clustering completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Batch clustering failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def find_similar_wallets(self, wallet_address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar wallets using similarity search"""
        try:
            return self.similarity_search.find_similar_wallets(wallet_address, limit)
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_cluster_overview(self) -> Dict[str, Any]:
        """Get overview of all clusters"""
        try:
            # Get all cluster centroids
            centroids = self.cluster_repo.get_all_cluster_centroids()
            
            # Get cluster statistics
            cluster_stats = []
            for centroid in centroids:
                stats = self.similarity_search.get_cluster_statistics(centroid.cluster_label)
                if 'error' not in stats:
                    cluster_stats.append(stats)
            
            # Calculate overall statistics
            total_wallets = sum(stats['wallet_count'] for stats in cluster_stats)
            
            return {
                'total_clusters': len(cluster_stats),
                'total_wallets': total_wallets,
                'clusters': cluster_stats,
                'overview_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster overview: {e}")
            return {'error': str(e)}
    
    def get_cluster_evolution(self, cluster_label: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get cluster evolution over time"""
        try:
            evolution_records = self.cluster_repo.get_cluster_evolution(cluster_label, days)
            
            # Convert to dictionary format
            evolution_data = []
            for record in evolution_records:
                evolution_data.append({
                    'cluster_label': record.cluster_label,
                    'cluster_size': record.cluster_size,
                    'avg_confidence': record.avg_confidence,
                    'stability_score': record.stability_score,
                    'recorded_at': record.recorded_at.isoformat()
                })
            
            return evolution_data
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster evolution: {e}")
            return []
    
    def _extract_wallet_features(self, wallet_address: str, wallet_data: Dict[str, Any]) -> WalletFeatures:
        """Extract wallet features from wallet data"""
        try:
            # Extract basic features
            portfolio_value = wallet_data.get('total_value_usd', 0.0)
            transaction_count = wallet_data.get('transaction_count', 0)
            
            # Calculate trading frequency (transactions per day over 30 days)
            trading_frequency = transaction_count / 30.0 if transaction_count > 0 else 0.0
            
            # Calculate average transaction size
            transactions = wallet_data.get('transactions', [])
            if transactions:
                transaction_amounts = [t.get('value_usd', 0) for t in transactions]
                avg_transaction_size = sum(transaction_amounts) / len(transaction_amounts)
            else:
                avg_transaction_size = 0.0
            
            # Calculate risk score (simplified)
            risk_score = self._calculate_risk_score(wallet_data)
            
            # Calculate protocol diversity
            protocol_diversity = self._calculate_protocol_diversity(transactions)
            
            return WalletFeatures(
                wallet_address=wallet_address,
                trading_frequency=trading_frequency,
                avg_transaction_size=avg_transaction_size,
                risk_score=risk_score,
                protocol_diversity=protocol_diversity,
                portfolio_value=portfolio_value,
                transaction_count=transaction_count,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract wallet features: {e}")
            # Return default features
            return WalletFeatures(
                wallet_address=wallet_address,
                trading_frequency=0.0,
                avg_transaction_size=0.0,
                risk_score=0.5,
                protocol_diversity=0.0,
                portfolio_value=0.0,
                transaction_count=0,
                last_updated=datetime.now()
            )
    
    def _calculate_risk_score(self, wallet_data: Dict[str, Any]) -> float:
        """Calculate risk score for wallet"""
        try:
            risk_factors = []
            
            # Portfolio concentration risk
            portfolio_value = wallet_data.get('total_value_usd', 0)
            if portfolio_value > 0:
                assets = wallet_data.get('assets', {})
                if assets:
                    # Calculate concentration (simplified)
                    asset_values = []
                    for chain_assets in assets.values():
                        for asset_data in chain_assets.values():
                            if isinstance(asset_data, dict) and 'value_usd' in asset_data:
                                asset_values.append(asset_data['value_usd'])
                    
                    if asset_values:
                        concentration = max(asset_values) / sum(asset_values)
                        risk_factors.append(concentration)
            
            # Trading frequency risk
            transaction_count = wallet_data.get('transaction_count', 0)
            if transaction_count > 100:
                risk_factors.append(0.8)  # High frequency trading
            elif transaction_count > 50:
                risk_factors.append(0.6)  # Moderate frequency
            else:
                risk_factors.append(0.3)  # Low frequency
            
            # Protocol diversity risk (lower diversity = higher risk)
            transactions = wallet_data.get('transactions', [])
            if transactions:
                protocols = set(t.get('protocol', '') for t in transactions if t.get('protocol'))
                diversity = len(protocols) / 10.0  # Normalize by max expected protocols
                risk_factors.append(1.0 - diversity)  # Inverse relationship
            
            # Calculate average risk score
            risk_score = sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
            return min(1.0, max(0.0, risk_score))  # Clamp to 0-1
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {e}")
            return 0.5
    
    def _calculate_protocol_diversity(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate protocol diversity score"""
        try:
            if not transactions:
                return 0.0
            
            protocols = set(t.get('protocol', '') for t in transactions if t.get('protocol'))
            diversity = len(protocols) / 10.0  # Normalize by max expected protocols
            return min(1.0, diversity)
            
        except Exception as e:
            self.logger.error(f"Protocol diversity calculation failed: {e}")
            return 0.0
    
    def _get_cluster_characteristics(self, cluster_assignment: Optional[ClusterAssignment]) -> Dict[str, Any]:
        """Get cluster characteristics"""
        try:
            if not cluster_assignment or cluster_assignment.cluster_label == -1:
                return {
                    'trading_style': 'unknown',
                    'risk_tolerance': 'unknown',
                    'protocol_preferences': [],
                    'typical_holdings': [],
                    'transaction_pattern': 'unknown'
                }
            
            # Get cluster centroid for characteristics
            centroid = self.cluster_repo.get_cluster_centroid(cluster_assignment.cluster_label)
            if not centroid:
                return {
                    'trading_style': 'unknown',
                    'risk_tolerance': 'unknown',
                    'protocol_preferences': [],
                    'typical_holdings': [],
                    'transaction_pattern': 'unknown'
                }
            
            # Determine characteristics based on centroid
            trading_freq = centroid.trading_frequency_centroid
            avg_size = centroid.avg_transaction_size_centroid
            risk_score = centroid.risk_score_centroid
            protocol_div = centroid.protocol_diversity_centroid
            
            # Trading style
            if trading_freq > 0.5:
                trading_style = 'active_trader'
            elif trading_freq > 0.2:
                trading_style = 'moderate_trader'
            else:
                trading_style = 'hodler'
            
            # Risk tolerance
            if risk_score > 0.7:
                risk_tolerance = 'high'
            elif risk_score < 0.3:
                risk_tolerance = 'low'
            else:
                risk_tolerance = 'medium'
            
            # Transaction pattern
            if avg_size > 5000:
                transaction_pattern = 'large_infrequent_trades'
            elif trading_freq > 0.3:
                transaction_pattern = 'frequent_small_trades'
            else:
                transaction_pattern = 'moderate_trading'
            
            return {
                'trading_style': trading_style,
                'risk_tolerance': risk_tolerance,
                'protocol_preferences': ['uniswap', 'aave', 'compound'],  # Default
                'typical_holdings': ['ETH', 'USDC', 'WBTC'],  # Default
                'transaction_pattern': transaction_pattern
            }
            
        except Exception as e:
            self.logger.error(f"Cluster characteristics calculation failed: {e}")
            return {
                'trading_style': 'unknown',
                'risk_tolerance': 'unknown',
                'protocol_preferences': [],
                'typical_holdings': [],
                'transaction_pattern': 'unknown'
            }
    
    def _get_default_clustering_result(self, wallet_address: str) -> Dict[str, Any]:
        """Get default clustering result"""
        return {
            'wallet_address': wallet_address,
            'cluster_label': -1,
            'cluster_confidence': 0.0,
            'distance_to_centroid': 0.0,
            'similar_wallets': [],
            'cluster_characteristics': {
                'trading_style': 'unknown',
                'risk_tolerance': 'unknown',
                'protocol_preferences': [],
                'typical_holdings': [],
                'transaction_pattern': 'unknown'
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
            'error': 'Wallet not found or clustering failed'
        }
    
    def close(self):
        """Close all connections"""
        try:
            # No explicit close needed for repositories as they manage their own connections
            if self.batch_clustering:
                self.batch_clustering.close()
            if self.similarity_search:
                self.similarity_search.close()
            if self.dtw_analyzer:
                self.dtw_analyzer.close()
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
