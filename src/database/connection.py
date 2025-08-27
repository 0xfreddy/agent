"""
Database connection manager for wallet clustering system
"""

import os
import logging
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from src.config import get_config

class DatabaseConnection:
    """Manages database connections for the clustering system"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            # Get database configuration
            db_config = {
                'host': self.config.database_host,
                'port': self.config.database_port,
                'database': self.config.database_name,
                'user': self.config.database_user,
                'password': self.config.database_password,
                'sslmode': 'require' if self.config.database_ssl else 'disable'
            }
            
            # Create connection pool
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                **db_config
            )
            
            self.logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool"""
        conn = None
        try:
            if self.pool:
                conn = self.pool.getconn()
                yield conn
            else:
                self.logger.error("Database pool not available")
                yield None
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            yield None
        finally:
            if conn and self.pool:
                self.pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = True):
        """Execute a database query"""
        with self.get_connection() as conn:
            if not conn:
                return None
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch:
                        if query.strip().upper().startswith('SELECT'):
                            return cursor.fetchall()
                        else:
                            conn.commit()
                            return cursor.rowcount
                    else:
                        conn.commit()
                        return cursor.rowcount
                        
            except Exception as e:
                self.logger.error(f"Query execution error: {e}")
                conn.rollback()
                return None
    
    def execute_many(self, query: str, params_list: list):
        """Execute multiple queries with different parameters"""
        with self.get_connection() as conn:
            if not conn:
                return None
            
            try:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params_list)
                    conn.commit()
                    return cursor.rowcount
                    
            except Exception as e:
                self.logger.error(f"Batch execution error: {e}")
                conn.rollback()
                return None
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        result = self.execute_query(query, (table_name,))
        return result[0]['exists'] if result else False
    
    def create_tables(self):
        """Create all required tables"""
        try:
            # Wallet features table
            wallet_features_table = """
            CREATE TABLE IF NOT EXISTS wallet_features (
                id SERIAL PRIMARY KEY,
                wallet_address VARCHAR(42) UNIQUE NOT NULL,
                trading_frequency DECIMAL(10,6),
                avg_transaction_size DECIMAL(20,2),
                risk_score DECIMAL(5,4),
                protocol_diversity DECIMAL(5,4),
                portfolio_value DECIMAL(20,2),
                transaction_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Cluster assignments table
            cluster_assignments_table = """
            CREATE TABLE IF NOT EXISTS cluster_assignments (
                id SERIAL PRIMARY KEY,
                wallet_address VARCHAR(42) UNIQUE NOT NULL,
                cluster_label INTEGER NOT NULL,
                cluster_confidence DECIMAL(5,4),
                distance_to_centroid DECIMAL(10,6),
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES wallet_features(wallet_address)
            );
            """
            
            # Cluster centroids table
            cluster_centroids_table = """
            CREATE TABLE IF NOT EXISTS cluster_centroids (
                id SERIAL PRIMARY KEY,
                cluster_label INTEGER UNIQUE NOT NULL,
                trading_frequency_centroid DECIMAL(10,6),
                avg_transaction_size_centroid DECIMAL(20,2),
                risk_score_centroid DECIMAL(5,4),
                protocol_diversity_centroid DECIMAL(5,4),
                cluster_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Cluster evolution table
            cluster_evolution_table = """
            CREATE TABLE IF NOT EXISTS cluster_evolution (
                id SERIAL PRIMARY KEY,
                cluster_label INTEGER NOT NULL,
                cluster_size INTEGER,
                avg_confidence DECIMAL(5,4),
                stability_score DECIMAL(5,4),
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Execute table creation
            self.execute_query(wallet_features_table, fetch=False)
            self.execute_query(cluster_assignments_table, fetch=False)
            self.execute_query(cluster_centroids_table, fetch=False)
            self.execute_query(cluster_evolution_table, fetch=False)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
    
    def close(self):
        """Close the database connection pool"""
        if self.pool:
            self.pool.closeall()
            self.logger.info("Database connection pool closed")
