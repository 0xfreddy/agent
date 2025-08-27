"""
ClickHouse database connection manager for wallet clustering system
"""

import logging
from typing import Optional, List, Dict, Any
import clickhouse_connect
from contextlib import contextmanager
from src.config import get_config

class ClickHouseConnection:
    """Manages ClickHouse database connections for the clustering system"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize ClickHouse connection"""
        try:
            # Get ClickHouse configuration
            self.client = clickhouse_connect.get_client(
                host=self.config.clickhouse_host,
                port=self.config.clickhouse_port,
                username=self.config.clickhouse_user,
                password=self.config.clickhouse_password,
                secure=self.config.clickhouse_secure,
                database=self.config.clickhouse_database
            )
            
            # Test connection
            result = self.client.query("SELECT 1")
            self.logger.info("ClickHouse connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ClickHouse connection: {e}")
            self.client = None
    
    @contextmanager
    def get_connection(self):
        """Get a ClickHouse connection"""
        try:
            if self.client:
                yield self.client
            else:
                self.logger.error("ClickHouse client not available")
                yield None
        except Exception as e:
            self.logger.error(f"ClickHouse connection error: {e}")
            yield None
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, fetch: bool = True):
        """Execute a ClickHouse query"""
        with self.get_connection() as client:
            if not client:
                return None
            
            try:
                if params:
                    result = client.query(query, parameters=params)
                else:
                    result = client.query(query)
                
                if fetch:
                    return result.result_set
                else:
                    return True
                        
            except Exception as e:
                self.logger.error(f"ClickHouse query execution error: {e}")
                return None
    
    def execute_many(self, query: str, params_list: List[Dict[str, Any]]):
        """Execute multiple queries with different parameters"""
        with self.get_connection() as client:
            if not client:
                return None
            
            try:
                client.insert(query, params_list)
                return len(params_list)
                    
            except Exception as e:
                self.logger.error(f"ClickHouse batch execution error: {e}")
                return None
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in ClickHouse"""
        query = """
        SELECT count() 
        FROM system.tables 
        WHERE database = %(database)s AND name = %(table)s
        """
        
        try:
            result = self.execute_query(
                query, 
                {'database': self.config.clickhouse_database, 'table': table_name}
            )
            return result[0][0] > 0 if result else False
        except Exception as e:
            self.logger.error(f"Table existence check failed: {e}")
            return False
    
    def create_tables(self):
        """Create all required tables in ClickHouse"""
        try:
            # Wallet features table
            wallet_features_table = """
            CREATE TABLE IF NOT EXISTS wallet_features (
                id UInt32,
                wallet_address String,
                trading_frequency Float64,
                avg_transaction_size Float64,
                risk_score Float64,
                protocol_diversity Float64,
                portfolio_value Float64,
                transaction_count UInt32,
                last_updated DateTime,
                created_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (wallet_address, created_at)
            """
            
            # Cluster assignments table
            cluster_assignments_table = """
            CREATE TABLE IF NOT EXISTS cluster_assignments (
                id UInt32,
                wallet_address String,
                cluster_label Int32,
                cluster_confidence Float64,
                distance_to_centroid Float64,
                assigned_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (wallet_address, assigned_at)
            """
            
            # Cluster centroids table
            cluster_centroids_table = """
            CREATE TABLE IF NOT EXISTS cluster_centroids (
                id UInt32,
                cluster_label Int32,
                trading_frequency_centroid Float64,
                avg_transaction_size_centroid Float64,
                risk_score_centroid Float64,
                protocol_diversity_centroid Float64,
                cluster_size UInt32,
                created_at DateTime,
                updated_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (cluster_label, updated_at)
            """
            
            # Cluster evolution table
            cluster_evolution_table = """
            CREATE TABLE IF NOT EXISTS cluster_evolution (
                id UInt32,
                cluster_label Int32,
                cluster_size UInt32,
                avg_confidence Float64,
                stability_score Float64,
                recorded_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY (cluster_label, recorded_at)
            """
            
            # Execute table creation
            self.execute_query(wallet_features_table, fetch=False)
            self.execute_query(cluster_assignments_table, fetch=False)
            self.execute_query(cluster_centroids_table, fetch=False)
            self.execute_query(cluster_evolution_table, fetch=False)
            
            self.logger.info("ClickHouse tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create ClickHouse tables: {e}")
    
    def close(self):
        """Close the ClickHouse connection"""
        if self.client:
            self.client.close()
            self.logger.info("ClickHouse connection closed")
