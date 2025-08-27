"""
Database models for wallet clustering system
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

@dataclass
class WalletFeatures:
    """Model for wallet behavioral features"""
    wallet_address: str
    trading_frequency: float
    avg_transaction_size: float
    risk_score: float
    protocol_diversity: float
    portfolio_value: float
    transaction_count: int
    last_updated: datetime
    created_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'wallet_address': self.wallet_address,
            'trading_frequency': self.trading_frequency,
            'avg_transaction_size': self.avg_transaction_size,
            'risk_score': self.risk_score,
            'protocol_diversity': self.protocol_diversity,
            'portfolio_value': self.portfolio_value,
            'transaction_count': self.transaction_count,
            'last_updated': self.last_updated,
            'created_at': self.created_at or datetime.now()
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy feature vector for clustering"""
        return np.array([
            self.trading_frequency,
            self.avg_transaction_size,
            self.risk_score,
            self.protocol_diversity
        ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalletFeatures':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            wallet_address=data['wallet_address'],
            trading_frequency=float(data['trading_frequency']),
            avg_transaction_size=float(data['avg_transaction_size']),
            risk_score=float(data['risk_score']),
            protocol_diversity=float(data['protocol_diversity']),
            portfolio_value=float(data['portfolio_value']),
            transaction_count=int(data['transaction_count']),
            last_updated=data['last_updated'],
            created_at=data.get('created_at')
        )

@dataclass
class ClusterAssignment:
    """Model for cluster assignments"""
    wallet_address: str
    cluster_label: int
    cluster_confidence: float
    distance_to_centroid: float
    assigned_at: datetime
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'wallet_address': self.wallet_address,
            'cluster_label': self.cluster_label,
            'cluster_confidence': self.cluster_confidence,
            'distance_to_centroid': self.distance_to_centroid,
            'assigned_at': self.assigned_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterAssignment':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            wallet_address=data['wallet_address'],
            cluster_label=int(data['cluster_label']),
            cluster_confidence=float(data['cluster_confidence']),
            distance_to_centroid=float(data['distance_to_centroid']),
            assigned_at=data['assigned_at']
        )

@dataclass
class ClusterCentroid:
    """Model for cluster centroids"""
    cluster_label: int
    trading_frequency_centroid: float
    avg_transaction_size_centroid: float
    risk_score_centroid: float
    protocol_diversity_centroid: float
    cluster_size: int
    created_at: datetime
    updated_at: datetime
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'cluster_label': self.cluster_label,
            'trading_frequency_centroid': self.trading_frequency_centroid,
            'avg_transaction_size_centroid': self.avg_transaction_size_centroid,
            'risk_score_centroid': self.risk_score_centroid,
            'protocol_diversity_centroid': self.protocol_diversity_centroid,
            'cluster_size': self.cluster_size,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    def to_centroid_vector(self) -> np.ndarray:
        """Convert to numpy centroid vector"""
        return np.array([
            self.trading_frequency_centroid,
            self.avg_transaction_size_centroid,
            self.risk_score_centroid,
            self.protocol_diversity_centroid
        ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterCentroid':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            cluster_label=int(data['cluster_label']),
            trading_frequency_centroid=float(data['trading_frequency_centroid']),
            avg_transaction_size_centroid=float(data['avg_transaction_size_centroid']),
            risk_score_centroid=float(data['risk_score_centroid']),
            protocol_diversity_centroid=float(data['protocol_diversity_centroid']),
            cluster_size=int(data['cluster_size']),
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )

@dataclass
class ClusterEvolution:
    """Model for tracking cluster evolution over time"""
    cluster_label: int
    cluster_size: int
    avg_confidence: float
    stability_score: float
    recorded_at: datetime
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'cluster_label': self.cluster_label,
            'cluster_size': self.cluster_size,
            'avg_confidence': self.avg_confidence,
            'stability_score': self.stability_score,
            'recorded_at': self.recorded_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterEvolution':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            cluster_label=int(data['cluster_label']),
            cluster_size=int(data['cluster_size']),
            avg_confidence=float(data['avg_confidence']),
            stability_score=float(data['stability_score']),
            recorded_at=data['recorded_at']
        )
