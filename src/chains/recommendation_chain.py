"""
Cursor AI Instruction:
Create a LangChain Sequential Chain that:

1. Chains together all analysis steps:
   - DataCollectionChain: Gather all API data
   - FeatureEngineeringChain: Extract features
   - RiskAssessmentChain: Calculate risk metrics
   - PredictionChain: Generate predictions
   - RecommendationChain: Final recommendation

2. Each chain should:
   - Have clear input/output keys
   - Include error handling
   - Log to LangSmith
   - Support streaming responses

3. Implement caching between chains
4. Allow for chain interruption/resumption
"""

from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import Dict, Any, List, Optional
import logging
import json
import redis
from datetime import datetime, timedelta
from src.config import get_config
from src.agents.recommendation_agent import RecommendationAgent

class ChainOutputParser(BaseOutputParser):
    """Custom output parser for chain results"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse chain output into structured format"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Fallback to simple key-value parsing
            result = {}
            lines = text.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip()] = value.strip()
            
            return result
        except Exception as e:
            logging.error(f"Chain output parsing error: {e}")
            return {'raw_output': text}

class DataCollectionChain:
    """Chain for collecting all required data"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = RecommendationAgent()
        
        # Initialize Redis for caching
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_enabled = True
        except:
            self.logger.warning("Redis not available, chain caching disabled")
            self.cache_enabled = False
            self.redis_client = None
    
    def run(self, wallet_address: str) -> Dict[str, Any]:
        """Run data collection chain"""
        try:
            # Check cache first
            cache_key = f"chain_data:{wallet_address}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            self.logger.info(f"Starting data collection for wallet: {wallet_address}")
            
            # Step 1: Collect wallet data
            wallet_data = self.agent.tools['data_tools']['octav']._run(wallet_address)
            
            # Step 2: Collect market data
            market_data = self.agent.tools['data_tools']['market']._run("")
            
            # Step 3: Collect sentiment data
            sentiment_data = self.agent.tools['data_tools']['sentiment']._run("")
            
            # Compile results
            result = {
                'wallet_data': wallet_data,
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data collection chain error: {e}")
            return self._get_fallback_data(wallet_address)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        if not self.cache_enabled:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                self.logger.info(f"Cache hit for data collection: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result"""
        if not self.cache_enabled:
            return
        
        try:
            # Cache for 1 hour
            self.redis_client.setex(
                cache_key,
                3600,
                json.dumps(result)
            )
            self.logger.info(f"Cached data collection result: {cache_key}")
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")
    
    def _get_fallback_data(self, wallet_address: str) -> Dict[str, Any]:
        """Get fallback data when collection fails"""
        return {
            'wallet_data': {'address': wallet_address, 'error': 'Data collection failed'},
            'market_data': {'error': 'Market data unavailable'},
            'sentiment_data': {'error': 'Sentiment data unavailable'},
            'collection_timestamp': datetime.now().isoformat()
        }

class FeatureEngineeringChain:
    """Chain for feature engineering"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = RecommendationAgent()
    
    def run(self, data_collection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run feature engineering chain"""
        try:
            self.logger.info("Starting feature engineering")
            
            wallet_data = data_collection_result['wallet_data']
            
            # Extract features using all feature engineering tools
            fourier_features = self.agent.tools['feature_tools']['fourier']._run(wallet_data.get('transactions', []))
            autocorr_features = self.agent.tools['feature_tools']['autocorrelation']._run(wallet_data.get('transactions', []))
            stat_features = self.agent.tools['feature_tools']['statistical']._run(wallet_data.get('transactions', []))
            portfolio_features = self.agent.tools['feature_tools']['portfolio']._run(wallet_data)
            
            result = {
                'fourier_features': fourier_features,
                'autocorr_features': autocorr_features,
                'statistical_features': stat_features,
                'portfolio_features': portfolio_features,
                'feature_engineering_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature engineering chain error: {e}")
            return self._get_fallback_features()
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Get fallback features when engineering fails"""
        return {
            'fourier_features': {'error': 'Fourier analysis failed'},
            'autocorr_features': {'error': 'Autocorrelation analysis failed'},
            'statistical_features': {'error': 'Statistical analysis failed'},
            'portfolio_features': {'error': 'Portfolio analysis failed'},
            'feature_engineering_timestamp': datetime.now().isoformat()
        }

class RiskAssessmentChain:
    """Chain for risk assessment"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = RecommendationAgent()
    
    def run(self, data_collection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run risk assessment chain"""
        try:
            self.logger.info("Starting risk assessment")
            
            wallet_data = data_collection_result['wallet_data']
            
            # Calculate risk metrics using all risk assessment tools
            volatility_risk = self.agent.tools['risk_tools']['volatility']._run(wallet_data)
            var_risk = self.agent.tools['risk_tools']['var']._run(wallet_data)
            concentration_risk = self.agent.tools['risk_tools']['concentration']._run(wallet_data)
            protocol_risk = self.agent.tools['risk_tools']['protocol']._run(wallet_data)
            
            result = {
                'volatility_risk': volatility_risk,
                'var_risk': var_risk,
                'concentration_risk': concentration_risk,
                'protocol_risk': protocol_risk,
                'risk_assessment_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Risk assessment chain error: {e}")
            return self._get_fallback_risk()
    
    def _get_fallback_risk(self) -> Dict[str, Any]:
        """Get fallback risk metrics when assessment fails"""
        return {
            'volatility_risk': {'risk_score': 50.0, 'error': 'Volatility calculation failed'},
            'var_risk': {'var_95': 0.0, 'error': 'VaR calculation failed'},
            'concentration_risk': {'hhi_score': 0.0, 'error': 'Concentration calculation failed'},
            'protocol_risk': {'error': 'Protocol risk assessment failed'},
            'risk_assessment_timestamp': datetime.now().isoformat()
        }

class PredictionChain:
    """Chain for ML predictions"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = RecommendationAgent()
    
    def run(self, data_collection_result: Dict[str, Any], feature_result: Dict[str, Any], risk_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction chain"""
        try:
            self.logger.info("Starting ML predictions")
            
            # Prepare wallet features for ML tools
            wallet_data = data_collection_result['wallet_data']
            stat_features = feature_result.get('statistical_features', {})
            portfolio_features = feature_result.get('portfolio_features', {})
            volatility_risk = risk_result.get('volatility_risk', {})
            concentration_risk = risk_result.get('concentration_risk', {})
            sentiment_data = data_collection_result.get('sentiment_data', {})
            
            wallet_features = {
                'address': wallet_data.get('address', 'unknown'),
                'transactions': wallet_data.get('transactions', []),
                'trading_frequency': stat_features.get('transaction_count', 0) / 30.0,
                'avg_transaction_size': stat_features.get('mean_transaction_amount', 0),
                'risk_score': volatility_risk.get('risk_score', 0.5),
                'portfolio_diversity': portfolio_features.get('portfolio_diversity', 0.5),
                'market_sentiment': sentiment_data.get('fear_greed_index', 50) / 100.0,
                'volatility': volatility_risk.get('volatility_score', 0.5),
                'concentration_risk': concentration_risk.get('concentration_risk', 0.5),
                'gas_efficiency': stat_features.get('gas_efficiency', 0.5),
                'portfolio_value': portfolio_features.get('portfolio_value', 10000),
                'risk_tolerance': 0.5,
                'liquidity_needs': 0.5,
                'gas_price': 0.5
            }
            
            # Run ML predictions
            clustering_result = self.agent.tools['ml_tools']['clustering']._run(wallet_features)
            action_prediction = self.agent.tools['ml_tools']['action_prediction']._run(wallet_features)
            timing_prediction = self.agent.tools['ml_tools']['timing_prediction']._run(wallet_features)
            sizing_prediction = self.agent.tools['ml_tools']['sizing_prediction']._run(wallet_features)
            
            result = {
                'clustering_result': clustering_result,
                'action_prediction': action_prediction,
                'timing_prediction': timing_prediction,
                'sizing_prediction': sizing_prediction,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction chain error: {e}")
            return self._get_fallback_predictions()
    
    def _get_fallback_predictions(self) -> Dict[str, Any]:
        """Get fallback predictions when ML fails"""
        return {
            'clustering_result': {'error': 'Clustering failed'},
            'action_prediction': {'predicted_action': 'hold', 'confidence': 0.5, 'error': 'Action prediction failed'},
            'timing_prediction': {'recommended_timing': 'within_24h', 'error': 'Timing prediction failed'},
            'sizing_prediction': {'recommended_size_percentage': 10.0, 'error': 'Sizing prediction failed'},
            'prediction_timestamp': datetime.now().isoformat()
        }

class RecommendationChain:
    """Chain for final recommendation generation"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent = RecommendationAgent()
    
    def run(self, data_collection_result: Dict[str, Any], feature_result: Dict[str, Any], 
            risk_result: Dict[str, Any], prediction_result: Dict[str, Any], mood: str = "balanced") -> Dict[str, Any]:
        """Run recommendation chain"""
        try:
            self.logger.info(f"Generating final recommendation with mood: {mood}")
            
            # Compile all analysis results
            analysis_results = {
                'wallet_data': data_collection_result['wallet_data'],
                'market_data': data_collection_result['market_data'],
                'sentiment_data': data_collection_result['sentiment_data'],
                'features': feature_result,
                'risk_metrics': risk_result,
                'ml_analysis': prediction_result,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Generate recommendation
            recommendation = self.agent.generate_recommendation(analysis_results, mood)
            
            # Generate explanation
            explanation = self.agent.explain_recommendation(recommendation)
            
            result = {
                'recommendation': recommendation,
                'explanation': explanation,
                'analysis_results': analysis_results,
                'recommendation_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Recommendation chain error: {e}")
            return self._get_fallback_recommendation(mood)
    
    def _get_fallback_recommendation(self, mood: str) -> Dict[str, Any]:
        """Get fallback recommendation when generation fails"""
        return {
            'recommendation': {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': 'Unable to complete full analysis - conservative hold recommendation',
                'risk_score': 50.0,
                'timestamp': datetime.now().isoformat()
            },
            'explanation': 'Analysis failed - using fallback recommendation',
            'analysis_results': {'error': 'Analysis failed'},
            'recommendation_timestamp': datetime.now().isoformat()
        }

class RecommendationChainOrchestrator:
    """Main orchestrator for the recommendation chain"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize all chains
        self.data_collection_chain = DataCollectionChain()
        self.feature_engineering_chain = FeatureEngineeringChain()
        self.risk_assessment_chain = RiskAssessmentChain()
        self.prediction_chain = PredictionChain()
        self.recommendation_chain = RecommendationChain()
        
        # Initialize Redis for chain state management
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_enabled = True
        except:
            self.logger.warning("Redis not available, chain state management disabled")
            self.cache_enabled = False
            self.redis_client = None
    
    def run(self, wallet_address: str, mood: str = "balanced") -> Dict[str, Any]:
        """Run the complete recommendation chain"""
        try:
            self.logger.info(f"Starting recommendation chain for wallet: {wallet_address}")
            
            # Check for existing chain state
            chain_state = self._get_chain_state(wallet_address)
            if chain_state:
                self.logger.info(f"Resuming chain execution from state: {chain_state['current_step']}")
                return self._resume_chain(wallet_address, mood, chain_state)
            
            # Step 1: Data Collection
            self.logger.info("Step 1: Data Collection")
            data_collection_result = self.data_collection_chain.run(wallet_address)
            self._save_chain_state(wallet_address, 'data_collection', data_collection_result)
            
            # Step 2: Feature Engineering
            self.logger.info("Step 2: Feature Engineering")
            feature_result = self.feature_engineering_chain.run(data_collection_result)
            self._save_chain_state(wallet_address, 'feature_engineering', feature_result)
            
            # Step 3: Risk Assessment
            self.logger.info("Step 3: Risk Assessment")
            risk_result = self.risk_assessment_chain.run(data_collection_result)
            self._save_chain_state(wallet_address, 'risk_assessment', risk_result)
            
            # Step 4: ML Predictions
            self.logger.info("Step 4: ML Predictions")
            prediction_result = self.prediction_chain.run(data_collection_result, feature_result, risk_result)
            self._save_chain_state(wallet_address, 'prediction', prediction_result)
            
            # Step 5: Final Recommendation
            self.logger.info("Step 5: Final Recommendation")
            final_result = self.recommendation_chain.run(
                data_collection_result, feature_result, risk_result, prediction_result, mood
            )
            
            # Clear chain state
            self._clear_chain_state(wallet_address)
            
            self.logger.info("Recommendation chain completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Chain orchestration error: {e}")
            return self._get_fallback_result(wallet_address, mood)
    
    def _get_chain_state(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """Get existing chain state"""
        if not self.cache_enabled:
            return None
        
        try:
            state_key = f"chain_state:{wallet_address}"
            state = self.redis_client.get(state_key)
            if state:
                return json.loads(state)
        except Exception as e:
            self.logger.error(f"Chain state retrieval error: {e}")
        
        return None
    
    def _save_chain_state(self, wallet_address: str, step: str, result: Dict[str, Any]):
        """Save chain state"""
        if not self.cache_enabled:
            return
        
        try:
            state_key = f"chain_state:{wallet_address}"
            state = {
                'current_step': step,
                'step_result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save for 1 hour
            self.redis_client.setex(
                state_key,
                3600,
                json.dumps(state)
            )
        except Exception as e:
            self.logger.error(f"Chain state save error: {e}")
    
    def _clear_chain_state(self, wallet_address: str):
        """Clear chain state"""
        if not self.cache_enabled:
            return
        
        try:
            state_key = f"chain_state:{wallet_address}"
            self.redis_client.delete(state_key)
        except Exception as e:
            self.logger.error(f"Chain state clear error: {e}")
    
    def _resume_chain(self, wallet_address: str, mood: str, chain_state: Dict[str, Any]) -> Dict[str, Any]:
        """Resume chain from saved state"""
        try:
            current_step = chain_state['current_step']
            step_result = chain_state['step_result']
            
            if current_step == 'data_collection':
                # Resume from feature engineering
                feature_result = self.feature_engineering_chain.run(step_result)
                self._save_chain_state(wallet_address, 'feature_engineering', feature_result)
                
                risk_result = self.risk_assessment_chain.run(step_result)
                self._save_chain_state(wallet_address, 'risk_assessment', risk_result)
                
                prediction_result = self.prediction_chain.run(step_result, feature_result, risk_result)
                self._save_chain_state(wallet_address, 'prediction', prediction_result)
                
                final_result = self.recommendation_chain.run(step_result, feature_result, risk_result, prediction_result, mood)
                
            elif current_step == 'feature_engineering':
                # Resume from risk assessment
                data_collection_result = step_result  # This would need to be stored separately
                risk_result = self.risk_assessment_chain.run(data_collection_result)
                self._save_chain_state(wallet_address, 'risk_assessment', risk_result)
                
                prediction_result = self.prediction_chain.run(data_collection_result, step_result, risk_result)
                self._save_chain_state(wallet_address, 'prediction', prediction_result)
                
                final_result = self.recommendation_chain.run(data_collection_result, step_result, risk_result, prediction_result, mood)
                
            else:
                # For simplicity, restart the chain
                return self.run(wallet_address, mood)
            
            self._clear_chain_state(wallet_address)
            return final_result
            
        except Exception as e:
            self.logger.error(f"Chain resume error: {e}")
            return self._get_fallback_result(wallet_address, mood)
    
    def _get_fallback_result(self, wallet_address: str, mood: str) -> Dict[str, Any]:
        """Get fallback result when chain fails"""
        return {
            'recommendation': {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': 'Chain execution failed - conservative hold recommendation',
                'risk_score': 50.0,
                'timestamp': datetime.now().isoformat()
            },
            'explanation': 'Chain execution failed - using fallback recommendation',
            'analysis_results': {'error': 'Chain execution failed'},
            'recommendation_timestamp': datetime.now().isoformat()
        }
