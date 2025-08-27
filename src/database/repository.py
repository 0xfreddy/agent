"""
Repository layer for database operations
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from .connection import DatabaseConnection
from .clickhouse_connection import ClickHouseConnection
from .models import WalletFeatures, ClusterAssignment, ClusterCentroid, ClusterEvolution
from src.config import get_config

class WalletRepository:
    """Repository for wallet features operations"""
    
    def __init__(self, db_connection=None):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize database connection based on configuration
        if self.config.use_clickhouse:
            self.db = ClickHouseConnection() if db_connection is None else db_connection
        else:
            self.db = DatabaseConnection() if db_connection is None else db_connection
    
    def store_wallet_features(self, wallet_features: WalletFeatures) -> bool:
        """Store or update wallet features"""
        try:
            if self.config.use_clickhouse:
                return self._store_wallet_features_clickhouse(wallet_features)
            else:
                return self._store_wallet_features_postgres(wallet_features)
            
        except Exception as e:
            self.logger.error(f"Failed to store wallet features: {e}")
            return False
    
    def _store_wallet_features_clickhouse(self, wallet_features: WalletFeatures) -> bool:
        """Store wallet features in ClickHouse"""
        try:
            # Check if wallet exists
            existing = self.get_wallet_features(wallet_features.wallet_address)
            
            if existing:
                # Delete existing record and reinsert (ClickHouse doesn't support UPDATE well)
                delete_query = """
                ALTER TABLE wallet_features 
                DELETE WHERE wallet_address = %(wallet_address)s
                """
                self.db.execute_query(delete_query, {'wallet_address': wallet_features.wallet_address}, fetch=False)
            
            # Insert new wallet (or reinsert after delete)
            query = """
            INSERT INTO wallet_features 
            (wallet_address, trading_frequency, avg_transaction_size, risk_score,
             protocol_diversity, portfolio_value, transaction_count, last_updated, created_at)
            VALUES (%(wallet_address)s, %(trading_frequency)s, %(avg_transaction_size)s, %(risk_score)s,
                    %(protocol_diversity)s, %(portfolio_value)s, %(transaction_count)s, %(last_updated)s, %(created_at)s)
            """
            
            params = {
                'wallet_address': wallet_features.wallet_address,
                'trading_frequency': wallet_features.trading_frequency,
                'avg_transaction_size': wallet_features.avg_transaction_size,
                'risk_score': wallet_features.risk_score,
                'protocol_diversity': wallet_features.protocol_diversity,
                'portfolio_value': wallet_features.portfolio_value,
                'transaction_count': wallet_features.transaction_count,
                'last_updated': datetime.now(),
                'created_at': wallet_features.created_at or datetime.now()
            }
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"ClickHouse wallet features storage failed: {e}")
            return False
    
    def _store_wallet_features_postgres(self, wallet_features: WalletFeatures) -> bool:
        """Store wallet features in PostgreSQL (original implementation)"""
        try:
            # Check if wallet exists
            existing = self.get_wallet_features(wallet_features.wallet_address)
            
            if existing:
                # Update existing wallet
                query = """
                UPDATE wallet_features 
                SET trading_frequency = %s, avg_transaction_size = %s, risk_score = %s,
                    protocol_diversity = %s, portfolio_value = %s, transaction_count = %s,
                    last_updated = %s
                WHERE wallet_address = %s
                """
                params = (
                    wallet_features.trading_frequency,
                    wallet_features.avg_transaction_size,
                    wallet_features.risk_score,
                    wallet_features.protocol_diversity,
                    wallet_features.portfolio_value,
                    wallet_features.transaction_count,
                    datetime.now(),
                    wallet_features.wallet_address
                )
            else:
                # Insert new wallet
                query = """
                INSERT INTO wallet_features 
                (wallet_address, trading_frequency, avg_transaction_size, risk_score,
                 protocol_diversity, portfolio_value, transaction_count, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    wallet_features.wallet_address,
                    wallet_features.trading_frequency,
                    wallet_features.avg_transaction_size,
                    wallet_features.risk_score,
                    wallet_features.protocol_diversity,
                    wallet_features.portfolio_value,
                    wallet_features.transaction_count,
                    datetime.now()
                )
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None and result > 0
            
        except Exception as e:
            self.logger.error(f"PostgreSQL wallet features storage failed: {e}")
            return False
    
    def get_wallet_features(self, wallet_address: str) -> Optional[WalletFeatures]:
        """Get wallet features by address"""
        try:
            if self.config.use_clickhouse:
                query = "SELECT * FROM wallet_features WHERE wallet_address = %(wallet_address)s ORDER BY last_updated DESC LIMIT 1"
                params = {'wallet_address': wallet_address}
            else:
                query = "SELECT * FROM wallet_features WHERE wallet_address = %s"
                params = (wallet_address,)
            
            result = self.db.execute_query(query, params)
            
            if result and len(result) > 0:
                row = result[0]
                if self.config.use_clickhouse:
                    # ClickHouse returns tuples, convert to dict-like structure
                    return WalletFeatures(
                        id=row[0],
                        wallet_address=row[1],
                        trading_frequency=row[2],
                        avg_transaction_size=row[3],
                        risk_score=row[4],
                        protocol_diversity=row[5],
                        portfolio_value=row[6],
                        transaction_count=row[7],
                        last_updated=row[8],
                        created_at=row[9] if len(row) > 9 else None
                    )
                else:
                    # PostgreSQL returns dict-like objects
                    return WalletFeatures.from_dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet features: {e}")
            return None
    
    def get_wallet_features_for_clustering(self) -> tuple:
        """Get wallet features and addresses for clustering"""
        try:
            if self.config.use_clickhouse:
                query = """
                SELECT wallet_address, trading_frequency, avg_transaction_size, 
                       risk_score, protocol_diversity
                FROM wallet_features 
                WHERE last_updated > %(cutoff_date)s
                """
                cutoff_date = datetime.now() - timedelta(days=7)
                params = {'cutoff_date': cutoff_date}
            else:
                query = """
                SELECT wallet_address, trading_frequency, avg_transaction_size, 
                       risk_score, protocol_diversity
                FROM wallet_features 
                WHERE last_updated > %s
                """
                cutoff_date = datetime.now() - timedelta(days=7)
                params = (cutoff_date,)
            
            result = self.db.execute_query(query, params)
            
            if not result:
                return [], []
            
            addresses = [row[0] for row in result]
            features = np.array([
                [row[1], row[2], row[3], row[4]]
                for row in result
            ])
            
            return addresses, features
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet features for clustering: {e}")
            return [], []
    
    def get_wallet_count(self) -> int:
        """Get total number of wallets"""
        try:
            query = "SELECT count() FROM wallet_features" if self.config.use_clickhouse else "SELECT COUNT(*) as count FROM wallet_features"
            result = self.db.execute_query(query)
            
            if self.config.use_clickhouse:
                return result[0][0] if result else 0
            else:
                return result[0]['count'] if result else 0
            
        except Exception as e:
            self.logger.error(f"Failed to get wallet count: {e}")
            return 0
    
    def get_all_wallet_features(self) -> List[WalletFeatures]:
        """Get all wallet features"""
        try:
            if self.config.use_clickhouse:
                query = """
                SELECT wallet_address, trading_frequency, avg_transaction_size, risk_score,
                       protocol_diversity, portfolio_value, transaction_count, last_updated, created_at
                FROM wallet_features
                ORDER BY created_at DESC
                """
            else:
                query = """
                SELECT wallet_address, trading_frequency, avg_transaction_size, risk_score,
                       protocol_diversity, portfolio_value, transaction_count, last_updated, created_at
                FROM wallet_features
                ORDER BY created_at DESC
                """
            
            result = self.db.execute_query(query)
            
            if not result:
                return []
            
            features = []
            for row in result:
                if self.config.use_clickhouse:
                    features.append(WalletFeatures(
                        wallet_address=row[0],
                        trading_frequency=float(row[1]),
                        avg_transaction_size=float(row[2]),
                        risk_score=float(row[3]),
                        protocol_diversity=float(row[4]),
                        portfolio_value=float(row[5]),
                        transaction_count=int(row[6]),
                        last_updated=row[7],
                        created_at=row[8]
                    ))
                else:
                    features.append(WalletFeatures(
                        wallet_address=row['wallet_address'],
                        trading_frequency=float(row['trading_frequency']),
                        avg_transaction_size=float(row['avg_transaction_size']),
                        risk_score=float(row['risk_score']),
                        protocol_diversity=float(row['protocol_diversity']),
                        portfolio_value=float(row['portfolio_value']),
                        transaction_count=int(row['transaction_count']),
                        last_updated=row['last_updated'],
                        created_at=row['created_at']
                    ))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to get all wallet features: {e}")
            return []

class ClusterRepository:
    """Repository for cluster operations"""
    
    def __init__(self, db_connection=None):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize database connection based on configuration
        if self.config.use_clickhouse:
            self.db = ClickHouseConnection() if db_connection is None else db_connection
        else:
            self.db = DatabaseConnection() if db_connection is None else db_connection
    
    def store_cluster_assignment(self, assignment: ClusterAssignment) -> bool:
        """Store or update cluster assignment"""
        try:
            if self.config.use_clickhouse:
                return self._store_cluster_assignment_clickhouse(assignment)
            else:
                return self._store_cluster_assignment_postgres(assignment)
            
        except Exception as e:
            self.logger.error(f"Failed to store cluster assignment: {e}")
            return False
    
    def _store_cluster_assignment_clickhouse(self, assignment: ClusterAssignment) -> bool:
        """Store cluster assignment in ClickHouse"""
        try:
            # Check if assignment exists
            existing = self.get_cluster_assignment(assignment.wallet_address)
            
            if existing:
                # Delete existing assignment and reinsert
                delete_query = """
                ALTER TABLE cluster_assignments 
                DELETE WHERE wallet_address = %(wallet_address)s
                """
                self.db.execute_query(delete_query, {'wallet_address': assignment.wallet_address}, fetch=False)
            
            # Insert new assignment (or reinsert after delete)
            query = """
            INSERT INTO cluster_assignments 
            (wallet_address, cluster_label, cluster_confidence, distance_to_centroid, assigned_at)
            VALUES (%(wallet_address)s, %(cluster_label)s, %(cluster_confidence)s, %(distance_to_centroid)s, %(assigned_at)s)
            """
            
            params = {
                'wallet_address': assignment.wallet_address,
                'cluster_label': assignment.cluster_label,
                'cluster_confidence': assignment.cluster_confidence,
                'distance_to_centroid': assignment.distance_to_centroid,
                'assigned_at': datetime.now()
            }
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"ClickHouse cluster assignment storage failed: {e}")
            return False
    
    def _store_cluster_assignment_postgres(self, assignment: ClusterAssignment) -> bool:
        """Store cluster assignment in PostgreSQL (original implementation)"""
        try:
            # Check if assignment exists
            existing = self.get_cluster_assignment(assignment.wallet_address)
            
            if existing:
                # Update existing assignment
                query = """
                UPDATE cluster_assignments 
                SET cluster_label = %s, cluster_confidence = %s, 
                    distance_to_centroid = %s, assigned_at = %s
                WHERE wallet_address = %s
                """
                params = (
                    assignment.cluster_label,
                    assignment.cluster_confidence,
                    assignment.distance_to_centroid,
                    datetime.now(),
                    assignment.wallet_address
                )
            else:
                # Insert new assignment
                query = """
                INSERT INTO cluster_assignments 
                (wallet_address, cluster_label, cluster_confidence, distance_to_centroid, assigned_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                params = (
                    assignment.wallet_address,
                    assignment.cluster_label,
                    assignment.cluster_confidence,
                    assignment.distance_to_centroid,
                    datetime.now()
                )
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None and result > 0
            
        except Exception as e:
            self.logger.error(f"PostgreSQL cluster assignment storage failed: {e}")
            return False
    
    def get_cluster_assignment(self, wallet_address: str) -> Optional[ClusterAssignment]:
        """Get cluster assignment by wallet address"""
        try:
            if self.config.use_clickhouse:
                query = "SELECT * FROM cluster_assignments WHERE wallet_address = %(wallet_address)s ORDER BY assigned_at DESC LIMIT 1"
                params = {'wallet_address': wallet_address}
            else:
                query = "SELECT * FROM cluster_assignments WHERE wallet_address = %s"
                params = (wallet_address,)
            
            result = self.db.execute_query(query, params)
            
            if result and len(result) > 0:
                row = result[0]
                if self.config.use_clickhouse:
                    # ClickHouse returns tuples, convert to dict-like structure
                    return ClusterAssignment(
                        id=row[0],
                        wallet_address=row[1],
                        cluster_label=row[2],
                        cluster_confidence=row[3],
                        distance_to_centroid=row[4],
                        assigned_at=row[5]
                    )
                else:
                    # PostgreSQL returns dict-like objects
                    return ClusterAssignment.from_dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster assignment: {e}")
            return None
    
    def get_wallets_by_cluster(self, cluster_label: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get wallets in a specific cluster"""
        try:
            if self.config.use_clickhouse:
                query = """
                SELECT ca.wallet_address, ca.cluster_confidence, ca.distance_to_centroid,
                       wf.trading_frequency, wf.avg_transaction_size, wf.risk_score, wf.protocol_diversity
                FROM cluster_assignments ca
                JOIN wallet_features wf ON ca.wallet_address = wf.wallet_address
                WHERE ca.cluster_label = %(cluster_label)s
                ORDER BY ca.cluster_confidence DESC, ca.distance_to_centroid ASC
                LIMIT %(limit)s
                """
                params = {'cluster_label': cluster_label, 'limit': limit}
            else:
                query = """
                SELECT ca.wallet_address, ca.cluster_confidence, ca.distance_to_centroid,
                       wf.trading_frequency, wf.avg_transaction_size, wf.risk_score, wf.protocol_diversity
                FROM cluster_assignments ca
                JOIN wallet_features wf ON ca.wallet_address = wf.wallet_address
                WHERE ca.cluster_label = %s
                ORDER BY ca.cluster_confidence DESC, ca.distance_to_centroid ASC
                LIMIT %s
                """
                params = (cluster_label, limit)
            
            result = self.db.execute_query(query, params)
            
            if self.config.use_clickhouse:
                # Convert ClickHouse tuples to dict format
                return [
                    {
                        'wallet_address': row[0],
                        'cluster_confidence': row[1],
                        'distance_to_centroid': row[2],
                        'trading_frequency': row[3],
                        'avg_transaction_size': row[4],
                        'risk_score': row[5],
                        'protocol_diversity': row[6]
                    }
                    for row in result
                ] if result else []
            else:
                return result if result else []
            
        except Exception as e:
            self.logger.error(f"Failed to get wallets by cluster: {e}")
            return []
    
    def store_cluster_centroid(self, centroid: ClusterCentroid) -> bool:
        """Store or update cluster centroid"""
        try:
            if self.config.use_clickhouse:
                return self._store_cluster_centroid_clickhouse(centroid)
            else:
                return self._store_cluster_centroid_postgres(centroid)
            
        except Exception as e:
            self.logger.error(f"Failed to store cluster centroid: {e}")
            return False
    
    def _store_cluster_centroid_clickhouse(self, centroid: ClusterCentroid) -> bool:
        """Store cluster centroid in ClickHouse"""
        try:
            # Check if centroid exists
            existing = self.get_cluster_centroid(centroid.cluster_label)
            
            if existing:
                # Update existing centroid
                query = """
                ALTER TABLE cluster_centroids 
                UPDATE trading_frequency_centroid = %(trading_frequency_centroid)s,
                       avg_transaction_size_centroid = %(avg_transaction_size_centroid)s,
                       risk_score_centroid = %(risk_score_centroid)s,
                       protocol_diversity_centroid = %(protocol_diversity_centroid)s,
                       cluster_size = %(cluster_size)s,
                       updated_at = %(updated_at)s
                WHERE cluster_label = %(cluster_label)s
                """
            else:
                # Insert new centroid
                query = """
                INSERT INTO cluster_centroids 
                (cluster_label, trading_frequency_centroid, avg_transaction_size_centroid,
                 risk_score_centroid, protocol_diversity_centroid, cluster_size, created_at, updated_at)
                VALUES (%(cluster_label)s, %(trading_frequency_centroid)s, %(avg_transaction_size_centroid)s,
                        %(risk_score_centroid)s, %(protocol_diversity_centroid)s, %(cluster_size)s, %(created_at)s, %(updated_at)s)
                """
            
            params = {
                'cluster_label': centroid.cluster_label,
                'trading_frequency_centroid': centroid.trading_frequency_centroid,
                'avg_transaction_size_centroid': centroid.avg_transaction_size_centroid,
                'risk_score_centroid': centroid.risk_score_centroid,
                'protocol_diversity_centroid': centroid.protocol_diversity_centroid,
                'cluster_size': centroid.cluster_size,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"ClickHouse cluster centroid storage failed: {e}")
            return False
    
    def _store_cluster_centroid_postgres(self, centroid: ClusterCentroid) -> bool:
        """Store cluster centroid in PostgreSQL (original implementation)"""
        try:
            # Check if centroid exists
            existing = self.get_cluster_centroid(centroid.cluster_label)
            
            if existing:
                # Update existing centroid
                query = """
                UPDATE cluster_centroids 
                SET trading_frequency_centroid = %s, avg_transaction_size_centroid = %s,
                    risk_score_centroid = %s, protocol_diversity_centroid = %s,
                    cluster_size = %s, updated_at = %s
                WHERE cluster_label = %s
                """
                params = (
                    centroid.trading_frequency_centroid,
                    centroid.avg_transaction_size_centroid,
                    centroid.risk_score_centroid,
                    centroid.protocol_diversity_centroid,
                    centroid.cluster_size,
                    datetime.now(),
                    centroid.cluster_label
                )
            else:
                # Insert new centroid
                query = """
                INSERT INTO cluster_centroids 
                (cluster_label, trading_frequency_centroid, avg_transaction_size_centroid,
                 risk_score_centroid, protocol_diversity_centroid, cluster_size, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    centroid.cluster_label,
                    centroid.trading_frequency_centroid,
                    centroid.avg_transaction_size_centroid,
                    centroid.risk_score_centroid,
                    centroid.protocol_diversity_centroid,
                    centroid.cluster_size,
                    datetime.now(),
                    datetime.now()
                )
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None and result > 0
            
        except Exception as e:
            self.logger.error(f"PostgreSQL cluster centroid storage failed: {e}")
            return False
    
    def get_cluster_centroid(self, cluster_label: int) -> Optional[ClusterCentroid]:
        """Get cluster centroid by label"""
        try:
            if self.config.use_clickhouse:
                query = "SELECT * FROM cluster_centroids WHERE cluster_label = %(cluster_label)s ORDER BY updated_at DESC LIMIT 1"
                params = {'cluster_label': cluster_label}
            else:
                query = "SELECT * FROM cluster_centroids WHERE cluster_label = %s"
                params = (cluster_label,)
            
            result = self.db.execute_query(query, params)
            
            if result and len(result) > 0:
                row = result[0]
                if self.config.use_clickhouse:
                    # ClickHouse returns tuples, convert to dict-like structure
                    return ClusterCentroid(
                        id=row[0],
                        cluster_label=row[1],
                        trading_frequency_centroid=row[2],
                        avg_transaction_size_centroid=row[3],
                        risk_score_centroid=row[4],
                        protocol_diversity_centroid=row[5],
                        cluster_size=row[6],
                        created_at=row[7],
                        updated_at=row[8]
                    )
                else:
                    # PostgreSQL returns dict-like objects
                    return ClusterCentroid.from_dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster centroid: {e}")
            return None
    
    def get_all_cluster_centroids(self) -> List[ClusterCentroid]:
        """Get all cluster centroids"""
        try:
            query = "SELECT * FROM cluster_centroids ORDER BY cluster_label"
            result = self.db.execute_query(query)
            
            if result:
                if self.config.use_clickhouse:
                    # Convert ClickHouse tuples to ClusterCentroid objects
                    return [
                        ClusterCentroid(
                            id=row[0],
                            cluster_label=row[1],
                            trading_frequency_centroid=row[2],
                            avg_transaction_size_centroid=row[3],
                            risk_score_centroid=row[4],
                            protocol_diversity_centroid=row[5],
                            cluster_size=row[6],
                            created_at=row[7],
                            updated_at=row[8]
                        )
                        for row in result
                    ]
                else:
                    return [ClusterCentroid.from_dict(row) for row in result]
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get all cluster centroids: {e}")
            return []
    
    def store_cluster_evolution(self, evolution: ClusterEvolution) -> bool:
        """Store cluster evolution record"""
        try:
            if self.config.use_clickhouse:
                query = """
                INSERT INTO cluster_evolution 
                (cluster_label, cluster_size, avg_confidence, stability_score, recorded_at)
                VALUES (%(cluster_label)s, %(cluster_size)s, %(avg_confidence)s, %(stability_score)s, %(recorded_at)s)
                """
                params = {
                    'cluster_label': evolution.cluster_label,
                    'cluster_size': evolution.cluster_size,
                    'avg_confidence': evolution.avg_confidence,
                    'stability_score': evolution.stability_score,
                    'recorded_at': datetime.now()
                }
            else:
                query = """
                INSERT INTO cluster_evolution 
                (cluster_label, cluster_size, avg_confidence, stability_score, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                params = (
                    evolution.cluster_label,
                    evolution.cluster_size,
                    evolution.avg_confidence,
                    evolution.stability_score,
                    datetime.now()
                )
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Failed to store cluster evolution: {e}")
            return False
    
    def get_cluster_evolution(self, cluster_label: int, days: int = 30) -> List[ClusterEvolution]:
        """Get cluster evolution over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if self.config.use_clickhouse:
                query = """
                SELECT * FROM cluster_evolution 
                WHERE cluster_label = %(cluster_label)s AND recorded_at > %(cutoff_date)s
                ORDER BY recorded_at ASC
                """
                params = {'cluster_label': cluster_label, 'cutoff_date': cutoff_date}
            else:
                query = """
                SELECT * FROM cluster_evolution 
                WHERE cluster_label = %s AND recorded_at > %s
                ORDER BY recorded_at ASC
                """
                params = (cluster_label, cutoff_date)
            
            result = self.db.execute_query(query, params)
            
            if result:
                if self.config.use_clickhouse:
                    # Convert ClickHouse tuples to ClusterEvolution objects
                    return [
                        ClusterEvolution(
                            id=row[0],
                            cluster_label=row[1],
                            cluster_size=row[2],
                            avg_confidence=row[3],
                            stability_score=row[4],
                            recorded_at=row[5]
                        )
                        for row in result
                    ]
                else:
                    return [ClusterEvolution.from_dict(row) for row in result]
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster evolution: {e}")
            return []
    
    def clear_old_cluster_assignments(self, days: int = 7) -> bool:
        """Clear old cluster assignments"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if self.config.use_clickhouse:
                query = """
                ALTER TABLE cluster_assignments 
                DELETE WHERE assigned_at < %(cutoff_date)s
                """
                params = {'cutoff_date': cutoff_date}
            else:
                query = """
                DELETE FROM cluster_assignments 
                WHERE assigned_at < %s
                """
                params = (cutoff_date,)
            
            result = self.db.execute_query(query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Failed to clear old cluster assignments: {e}")
            return False
