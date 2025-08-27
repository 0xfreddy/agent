"""
Production monitoring utilities for clustering system
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from src.config import get_config

@dataclass
class ClusteringMetrics:
    """Metrics for clustering performance"""
    wallet_count: int
    cluster_count: int
    noise_count: int
    silhouette_score: float
    calinski_harabasz_score: float
    processing_time_seconds: float
    memory_usage_mb: float
    error_count: int
    timestamp: datetime

class MetricsTracker:
    """Track clustering performance metrics"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history: List[ClusteringMetrics] = []
        self.start_time: Optional[float] = None
    
    def start_tracking(self):
        """Start tracking a clustering operation"""
        self.start_time = time.time()
        self.logger.info("Started clustering metrics tracking")
    
    def record_metrics(self, 
                      wallet_count: int,
                      cluster_count: int,
                      noise_count: int,
                      silhouette_score: float,
                      calinski_harabasz_score: float,
                      error_count: int = 0) -> ClusteringMetrics:
        """Record clustering metrics"""
        if self.start_time is None:
            self.logger.warning("Metrics tracking not started")
            return None
        
        processing_time = time.time() - self.start_time
        memory_usage = self._get_memory_usage()
        
        metrics = ClusteringMetrics(
            wallet_count=wallet_count,
            cluster_count=cluster_count,
            noise_count=noise_count,
            silhouette_score=silhouette_score,
            calinski_harabasz_score=calinski_harabasz_score,
            processing_time_seconds=processing_time,
            memory_usage_mb=memory_usage,
            error_count=error_count,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _log_metrics(self, metrics: ClusteringMetrics):
        """Log metrics to monitoring system"""
        self.logger.info(f"Clustering metrics: {asdict(metrics)}")
        
        # Log to external monitoring if configured
        if hasattr(self.config, 'monitoring_endpoint'):
            self._send_to_monitoring(metrics)
    
    def _send_to_monitoring(self, metrics: ClusteringMetrics):
        """Send metrics to external monitoring system"""
        try:
            # This would integrate with your monitoring system (e.g., DataDog, Prometheus)
            pass
        except Exception as e:
            self.logger.error(f"Failed to send metrics to monitoring: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_date]
        
        if not recent_metrics:
            return {'error': 'No metrics available'}
        
        return {
            'total_runs': len(recent_metrics),
            'avg_processing_time': np.mean([m.processing_time_seconds for m in recent_metrics]),
            'avg_silhouette_score': np.mean([m.silhouette_score for m in recent_metrics]),
            'avg_cluster_count': np.mean([m.cluster_count for m in recent_metrics]),
            'total_errors': sum([m.error_count for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage_mb for m in recent_metrics])
        }

class DriftDetector:
    """Detect data drift in clustering features"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reference_distribution: Optional[Dict[str, np.ndarray]] = None
        self.psi_threshold = self.config.psi_threshold
    
    def set_reference_distribution(self, features: np.ndarray, feature_names: List[str]):
        """Set reference distribution for drift detection"""
        self.reference_distribution = {}
        for i, name in enumerate(feature_names):
            self.reference_distribution[name] = features[:, i]
        
        self.logger.info(f"Set reference distribution for {len(feature_names)} features")
    
    def calculate_psi(self, current_features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate Population Stability Index for each feature"""
        if self.reference_distribution is None:
            self.logger.warning("No reference distribution set")
            return {}
        
        psi_scores = {}
        
        for i, name in enumerate(feature_names):
            if name in self.reference_distribution:
                psi = self._calculate_feature_psi(
                    self.reference_distribution[name],
                    current_features[:, i]
                )
                psi_scores[name] = psi
        
        return psi_scores
    
    def _calculate_feature_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI for a single feature"""
        try:
            # Create histograms
            ref_hist, ref_edges = np.histogram(reference, bins=10, density=True)
            curr_hist, curr_edges = np.histogram(current, bins=ref_edges, density=True)
            
            # Avoid division by zero
            ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
            curr_hist = np.where(curr_hist == 0, 0.0001, curr_hist)
            
            # Calculate PSI
            psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
            
            return float(psi)
            
        except Exception as e:
            self.logger.error(f"PSI calculation failed: {e}")
            return 0.0
    
    def detect_drift(self, current_features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect data drift in features"""
        psi_scores = self.calculate_psi(current_features, feature_names)
        
        # Identify features with significant drift
        drifted_features = {
            name: score for name, score in psi_scores.items() 
            if score > self.psi_threshold
        }
        
        return {
            'psi_scores': psi_scores,
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0,
            'timestamp': datetime.now().isoformat()
        }

class LangSmithLogger:
    """Enhanced logging for LangSmith integration"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def log_clustering_run(self, 
                          wallet_address: str,
                          clustering_result: Dict[str, Any],
                          processing_time: float,
                          success: bool):
        """Log clustering run to LangSmith"""
        try:
            # This would integrate with LangSmith for detailed run tracking
            run_data = {
                'wallet_address': wallet_address,
                'cluster_label': clustering_result.get('cluster_label', -1),
                'cluster_confidence': clustering_result.get('cluster_confidence', 0.0),
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"LangSmith run logged: {run_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log to LangSmith: {e}")
    
    def log_batch_clustering(self, 
                           batch_result: Dict[str, Any],
                           metrics: Optional[ClusteringMetrics] = None):
        """Log batch clustering results"""
        try:
            batch_data = {
                'status': batch_result.get('status'),
                'wallet_count': batch_result.get('wallet_count', 0),
                'cluster_count': batch_result.get('cluster_count', 0),
                'noise_count': batch_result.get('noise_count', 0),
                'metrics': asdict(metrics) if metrics else None,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Batch clustering logged: {batch_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log batch clustering: {e}")

class PerformanceMonitor:
    """Monitor clustering system performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_tracker = MetricsTracker()
        self.drift_detector = DriftDetector()
        self.langsmith_logger = LangSmithLogger()
    
    def monitor_clustering_performance(self, 
                                     wallet_address: str,
                                     clustering_result: Dict[str, Any],
                                     processing_time: float,
                                     success: bool):
        """Monitor individual clustering performance"""
        # Log to LangSmith
        self.langsmith_logger.log_clustering_run(
            wallet_address, clustering_result, processing_time, success
        )
        
        # Track metrics if this is a batch operation
        if success and 'wallet_count' in clustering_result:
            self.metrics_tracker.record_metrics(
                wallet_count=clustering_result.get('wallet_count', 0),
                cluster_count=clustering_result.get('cluster_count', 0),
                noise_count=clustering_result.get('noise_count', 0),
                silhouette_score=clustering_result.get('metrics', {}).get('silhouette_score', 0.0),
                calinski_harabasz_score=clustering_result.get('metrics', {}).get('calinski_harabasz_score', 0.0)
            )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            performance_summary = self.metrics_tracker.get_performance_summary()
            
            return {
                'status': 'healthy',
                'performance_summary': performance_summary,
                'last_check': datetime.now().isoformat(),
                'drift_detection_enabled': self.drift_detector.reference_distribution is not None
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
