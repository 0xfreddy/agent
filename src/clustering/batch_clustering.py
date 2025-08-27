"""
Batch clustering job for daily wallet clustering
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from src.database import WalletRepository, ClusterRepository
from src.database.models import WalletFeatures, ClusterAssignment, ClusterCentroid, ClusterEvolution
from src.config import get_config

class BatchClusteringJob:
    """Daily batch clustering job"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize repositories (they handle database connection internally)
        self.wallet_repo = WalletRepository()
        self.cluster_repo = ClusterRepository()
        
        # Clustering parameters
        self.min_samples = self.config.cluster_min_samples
        self.eps = self.config.cluster_eps
        
        # Feature weights for clustering
        self.feature_weights = {
            'trading_frequency': 0.3,
            'avg_transaction_size': 0.25,
            'risk_score': 0.25,
            'protocol_diversity': 0.2
        }
    
    def run(self) -> Dict[str, Any]:
        """Run batch clustering job (alias for run_daily_clustering)"""
        return self.run_daily_clustering()
    
    def run_daily_clustering(self) -> Dict[str, Any]:
        """Run daily batch clustering job"""
        try:
            self.logger.info("Starting daily batch clustering job")
            
            # Step 1: Get wallet features for clustering
            addresses, features = self.wallet_repo.get_wallet_features_for_clustering()
            
            if len(addresses) < self.min_samples:
                self.logger.warning(f"Insufficient wallets for clustering: {len(addresses)} < {self.min_samples}")
                return {'status': 'insufficient_data', 'wallet_count': len(addresses)}
            
            self.logger.info(f"Clustering {len(addresses)} wallets")
            
            # Step 2: Preprocess features
            processed_features = self._preprocess_features(features)
            
            # Step 3: Run DBSCAN clustering
            cluster_labels = self._run_dbscan_clustering(processed_features)
            
            # Step 4: Calculate cluster metrics
            cluster_metrics = self._calculate_cluster_metrics(processed_features, cluster_labels)
            
            # Step 5: Calculate cluster centroids
            centroids = self._calculate_cluster_centroids(processed_features, cluster_labels)
            
            # Step 6: Store cluster assignments
            self._store_cluster_assignments(addresses, processed_features, cluster_labels)
            
            # Step 7: Store cluster centroids
            self._store_cluster_centroids(centroids, cluster_metrics)
            
            # Step 8: Track cluster evolution
            self._track_cluster_evolution(cluster_metrics)
            
            # Step 9: Clean up old assignments
            self.cluster_repo.clear_old_cluster_assignments(days=7)
            
            self.logger.info("Daily batch clustering completed successfully")
            
            return {
                'status': 'success',
                'wallet_count': len(addresses),
                'cluster_count': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'noise_count': list(cluster_labels).count(-1),
                'metrics': cluster_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Batch clustering failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features for clustering"""
        try:
            # Apply feature weights
            weighted_features = features * np.array(list(self.feature_weights.values()))
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(weighted_features)
            
            # Clip to reasonable range
            processed_features = np.clip(scaled_features, -3, 3)
            
            self.logger.info(f"Preprocessed {features.shape[0]} feature vectors")
            return processed_features
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            return features
    
    def _run_dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Run DBSCAN clustering"""
        try:
            # Initialize DBSCAN
            dbscan = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='euclidean',
                n_jobs=-1
            )
            
            # Fit and predict
            cluster_labels = dbscan.fit_predict(features)
            
            # Log clustering results
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            self.logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"DBSCAN clustering failed: {e}")
            # Return all noise labels as fallback
            return np.full(features.shape[0], -1)
    
    def _calculate_cluster_metrics(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        try:
            # Remove noise points for metric calculation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
            
            features_clean = features[non_noise_mask]
            labels_clean = cluster_labels[non_noise_mask]
            
            # Calculate metrics
            silhouette = silhouette_score(features_clean, labels_clean) if len(set(labels_clean)) > 1 else 0.0
            calinski_harabasz = calinski_harabasz_score(features_clean, labels_clean) if len(set(labels_clean)) > 1 else 0.0
            
            # Calculate cluster sizes
            unique_labels = set(cluster_labels)
            cluster_sizes = {label: list(cluster_labels).count(label) for label in unique_labels}
            
            # Calculate average confidence (distance-based)
            avg_confidences = {}
            for label in unique_labels:
                if label == -1:
                    continue
                mask = cluster_labels == label
                distances = np.linalg.norm(features[mask] - np.mean(features[mask], axis=0), axis=1)
                avg_confidences[label] = 1.0 / (1.0 + np.mean(distances))
            
            return {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski_harabasz),
                'cluster_sizes': cluster_sizes,
                'avg_confidences': avg_confidences,
                'total_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
                'noise_count': cluster_sizes.get(-1, 0)
            }
            
        except Exception as e:
            self.logger.error(f"Cluster metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
    
    def _calculate_cluster_centroids(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate cluster centroids"""
        try:
            centroids = {}
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise cluster
                    continue
                
                mask = cluster_labels == label
                cluster_features = features[mask]
                centroid = np.mean(cluster_features, axis=0)
                centroids[label] = centroid
            
            self.logger.info(f"Calculated centroids for {len(centroids)} clusters")
            return centroids
            
        except Exception as e:
            self.logger.error(f"Centroid calculation failed: {e}")
            return {}
    
    def _store_cluster_assignments(self, addresses: List[str], features: np.ndarray, cluster_labels: np.ndarray):
        """Store cluster assignments in database"""
        try:
            assignments = []
            
            for i, (address, label) in enumerate(zip(addresses, cluster_labels)):
                # Calculate distance to centroid
                distance_to_centroid = 0.0
                if label != -1:
                    # Get centroid for this cluster
                    centroid = self._get_cluster_centroid(features, cluster_labels, label)
                    if centroid is not None:
                        distance_to_centroid = np.linalg.norm(features[i] - centroid)
                
                # Calculate confidence based on distance
                confidence = 1.0 / (1.0 + distance_to_centroid) if label != -1 else 0.0
                
                assignment = ClusterAssignment(
                    wallet_address=address,
                    cluster_label=int(label),
                    cluster_confidence=float(confidence),
                    distance_to_centroid=float(distance_to_centroid),
                    assigned_at=datetime.now()
                )
                
                assignments.append(assignment)
            
            # Store assignments
            success_count = 0
            for assignment in assignments:
                if self.cluster_repo.store_cluster_assignment(assignment):
                    success_count += 1
            
            self.logger.info(f"Stored {success_count}/{len(assignments)} cluster assignments")
            
        except Exception as e:
            self.logger.error(f"Failed to store cluster assignments: {e}")
    
    def _store_cluster_centroids(self, centroids: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        """Store cluster centroids in database"""
        try:
            for label, centroid in centroids.items():
                # Convert centroid back to original feature space
                original_centroid = centroid / np.array(list(self.feature_weights.values()))
                
                cluster_centroid = ClusterCentroid(
                    cluster_label=label,
                    trading_frequency_centroid=float(original_centroid[0]),
                    avg_transaction_size_centroid=float(original_centroid[1]),
                    risk_score_centroid=float(original_centroid[2]),
                    protocol_diversity_centroid=float(original_centroid[3]),
                    cluster_size=metrics['cluster_sizes'].get(label, 0),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self.cluster_repo.store_cluster_centroid(cluster_centroid)
            
            self.logger.info(f"Stored {len(centroids)} cluster centroids")
            
        except Exception as e:
            self.logger.error(f"Failed to store cluster centroids: {e}")
    
    def _track_cluster_evolution(self, metrics: Dict[str, Any]):
        """Track cluster evolution over time"""
        try:
            for label, size in metrics['cluster_sizes'].items():
                if label == -1:  # Skip noise cluster
                    continue
                
                avg_confidence = metrics['avg_confidences'].get(label, 0.0)
                
                # Calculate stability score (simple heuristic)
                stability_score = min(1.0, size / 100.0) * avg_confidence
                
                evolution = ClusterEvolution(
                    cluster_label=label,
                    cluster_size=size,
                    avg_confidence=avg_confidence,
                    stability_score=stability_score,
                    recorded_at=datetime.now()
                )
                
                self.cluster_repo.store_cluster_evolution(evolution)
            
            self.logger.info("Tracked cluster evolution")
            
        except Exception as e:
            self.logger.error(f"Failed to track cluster evolution: {e}")
    
    def _get_cluster_centroid(self, features: np.ndarray, cluster_labels: np.ndarray, label: int) -> Optional[np.ndarray]:
        """Get centroid for a specific cluster"""
        try:
            mask = cluster_labels == label
            if np.sum(mask) == 0:
                return None
            
            cluster_features = features[mask]
            return np.mean(cluster_features, axis=0)
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster centroid: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        # No explicit close needed here as repositories handle their own connections
        pass
