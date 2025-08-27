from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent
from src.tools.data_collection import OctavDataTool, TVLDataTool, MarketDataTool, SentimentDataTool
from src.tools.feature_engineering import (
    FourierTransformTool, 
    AutocorrelationTool, 
    StatisticalFeatureTool,
    PortfolioFeatureTool
)
from src.tools.risk_assessment import (
    PortfolioVolatilityTool,
    VaRCalculatorTool,
    ConcentrationRiskTool,
    ProtocolRiskTool
)
from src.tools.clustering import WalletClusteringTool
from src.tools.ml_models import ActionPredictionTool, TimingPredictionTool, SizingPredictionTool
from src.models.schemas import Mood, Action, RiskCategory
import logging
from datetime import datetime
import numpy as np

class RecommendationAgent(BaseAgent):
    def __init__(self):
        super().__init__(project_name="crypto-recommendation")
        self.tools = self._initialize_tools()
        self.mood_settings = {
            "degen": {"risk_multiplier": 1.5, "confidence_threshold": 0.5, "aggression": 0.8},
            "balanced": {"risk_multiplier": 1.0, "confidence_threshold": 0.65, "aggression": 0.5},
            "saver": {"risk_multiplier": 0.5, "confidence_threshold": 0.8, "aggression": 0.2}
        }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all analysis tools"""
        return {
            'data_tools': {
                'octav': OctavDataTool(),
                'market': MarketDataTool(),
                'sentiment': SentimentDataTool()
            },
            'feature_tools': {
                'fourier': FourierTransformTool(),
                'autocorrelation': AutocorrelationTool(),
                'statistical': StatisticalFeatureTool(),
                'portfolio': PortfolioFeatureTool()
            },
            'risk_tools': {
                'volatility': PortfolioVolatilityTool(),
                'var': VaRCalculatorTool(),
                'concentration': ConcentrationRiskTool(),
                'protocol': ProtocolRiskTool()
            },
            'ml_tools': {
                'clustering': WalletClusteringTool(),
                'action_prediction': ActionPredictionTool(),
                'timing_prediction': TimingPredictionTool(),
                'sizing_prediction': SizingPredictionTool()
            }
        }
    
    def analyze(self, wallet_address: str, mood: str = "balanced") -> Dict[str, Any]:
        """Full wallet analysis pipeline"""
        try:
            self.logger.info(f"Starting analysis for wallet {wallet_address} with mood {mood}")
            
            # Step 1: Data Collection
            wallet_data = self.tools['data_tools']['octav']._run(wallet_address)
            market_data = self.tools['data_tools']['market']._run("")
            sentiment_data = self.tools['data_tools']['sentiment']._run("")
            
            # Step 2: Feature Engineering
            fourier_features = self.tools['feature_tools']['fourier']._run(wallet_data['transactions'])
            autocorr_features = self.tools['feature_tools']['autocorrelation']._run(wallet_data['transactions'])
            stat_features = self.tools['feature_tools']['statistical']._run(wallet_data['transactions'])
            portfolio_features = self.tools['feature_tools']['portfolio']._run(wallet_data)
            
            # Step 3: Risk Assessment
            volatility_risk = self.tools['risk_tools']['volatility']._run(wallet_data)
            var_risk = self.tools['risk_tools']['var']._run(wallet_data)
            concentration_risk = self.tools['risk_tools']['concentration']._run(wallet_data)
            
            # Step 4: ML Analysis
            # Prepare wallet features for ML tools
            wallet_features = {
                'address': wallet_address,
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
                'risk_tolerance': 0.5,  # Default, could be user-provided
                'liquidity_needs': 0.5,  # Default
                'gas_price': 0.5  # Default
            }
            
            # Run ML tools
            clustering_result = self.tools['ml_tools']['clustering']._run(wallet_features)
            action_prediction = self.tools['ml_tools']['action_prediction']._run(wallet_features)
            timing_prediction = self.tools['ml_tools']['timing_prediction']._run(wallet_features)
            sizing_prediction = self.tools['ml_tools']['sizing_prediction']._run(wallet_features)
            
            # Step 5: Compile Analysis Results
            analysis_results = {
                'wallet_data': wallet_data,
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'features': {
                    'fourier': fourier_features,
                    'autocorrelation': autocorr_features,
                    'statistical': stat_features,
                    'portfolio': portfolio_features
                },
                'risk_metrics': {
                    'volatility': volatility_risk,
                    'var': var_risk,
                    'concentration': concentration_risk
                },
                'ml_analysis': {
                    'clustering': clustering_result,
                    'action_prediction': action_prediction,
                    'timing_prediction': timing_prediction,
                    'sizing_prediction': sizing_prediction
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            self.track_metrics('analysis_complete', {
                'success': True,
                'wallet_address': wallet_address,
                'mood': mood,
                'transaction_count': len(wallet_data['transactions']),
                'total_value': wallet_data['total_value_usd']
            })
            
            return analysis_results
            
        except Exception as e:
            self.log_error(e, {'wallet_address': wallet_address, 'mood': mood})
            self.track_metrics('analysis_complete', {'success': False, 'error': str(e)})
            raise
    
    def generate_recommendation(self, analysis_results: Dict[str, Any], mood: str = "balanced") -> Dict[str, Any]:
        """Generate trading recommendation based on analysis"""
        try:
            mood_config = self.mood_settings.get(mood, self.mood_settings['balanced'])
            
            # Extract key metrics
            wallet_data = analysis_results['wallet_data']
            risk_metrics = analysis_results['risk_metrics']
            features = analysis_results['features']
            ml_analysis = analysis_results.get('ml_analysis', {})
            
            # Calculate risk-adjusted metrics
            base_risk_score = risk_metrics.get('volatility_risk', {}).get('volatility_score', 50.0)
            adjusted_risk_score = base_risk_score * mood_config['risk_multiplier']
            
            # Extract ML predictions
            action_prediction = ml_analysis.get('action_prediction', {})
            timing_prediction = ml_analysis.get('timing_prediction', {})
            sizing_prediction = ml_analysis.get('sizing_prediction', {})
            clustering_result = ml_analysis.get('clustering', {})
            
            # Determine action based on analysis and ML predictions
            action, confidence, reasoning = self._determine_action_with_ml(
                wallet_data, risk_metrics, features, mood_config, action_prediction
            )
            
            # Generate recommendation
            recommendation = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'risk_score': adjusted_risk_score,
                'risk_category': self._categorize_risk(adjusted_risk_score),
                'mood_adjustment': mood_config,
                'ml_insights': {
                    'predicted_action': action_prediction.get('predicted_action', 'hold'),
                    'action_confidence': action_prediction.get('confidence', 0.5),
                    'recommended_timing': timing_prediction.get('recommended_timing', 'within_24h'),
                    'recommended_size_percentage': sizing_prediction.get('recommended_size_percentage', 10.0),
                    'cluster_characteristics': clustering_result.get('cluster_characteristics', {}),
                    'similar_wallets': clustering_result.get('similar_wallets', [])
                },
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'portfolio_value': wallet_data.get('total_value_usd', 0),
                    'transaction_count': wallet_data.get('transaction_count', 0),
                    'volatility': risk_metrics.get('volatility_risk', {}).get('volatility_percentage', 0),
                    'var_amount': risk_metrics.get('var_risk', {}).get('var_95_dollar', 0),
                    'concentration_risk': risk_metrics.get('concentration_risk', {}).get('hhi_score', 0)
                }
            }
            
            self.track_metrics('recommendation_generated', {
                'action': action,
                'confidence': confidence,
                'risk_score': adjusted_risk_score,
                'mood': mood
            })
            
            return recommendation
            
        except Exception as e:
            self.log_error(e, {'analysis_results': analysis_results, 'mood': mood})
            return self._get_fallback_recommendation()
    
    def _determine_action_with_ml(self, wallet_data: Dict, risk_metrics: Dict, features: Dict, mood_config: Dict, action_prediction: Dict) -> tuple:
        """Determine the recommended action based on analysis and ML predictions"""
        
        # Extract key indicators
        volatility = risk_metrics.get('volatility_risk', {}).get('volatility_score', 50.0)
        var_amount = risk_metrics.get('var_risk', {}).get('var_95_dollar', 0)
        concentration = risk_metrics.get('concentration_risk', {}).get('hhi_score', 0)
        
        # Trading pattern analysis
        fourier_pattern = features.get('fourier_features', {}).get('pattern_type', 'unknown')
        stat_features = features.get('statistical_features', {})
        
        # ML prediction
        ml_action = action_prediction.get('predicted_action', 'hold')
        ml_confidence = action_prediction.get('confidence', 0.5)
        
        # Combine ML prediction with traditional analysis
        # Weight ML prediction based on confidence and mood
        ml_weight = min(ml_confidence * 0.7, 0.5)  # Max 50% weight for ML
        traditional_weight = 1 - ml_weight
        
        # Decision logic based on mood with ML integration
        if mood_config['aggression'] > 0.7:  # Degen mode
            if ml_action == 'swap' and ml_confidence > 0.6:
                return Action.SWAP, 0.75, f"ML predicts {ml_action} with {ml_confidence:.1%} confidence - aggressive trading opportunity"
            elif volatility < 30 and concentration < 0.3:
                return Action.SWAP, 0.75, "Low volatility and good diversification - good time for aggressive trading"
            elif stat_features['whale_transaction_ratio'] > 0.5:
                return Action.ADD_LIQUIDITY, 0.65, "High whale activity detected - liquidity provision opportunity"
            else:
                return Action.HOLD, 0.6, "High risk environment - hold and wait for better opportunities"
        
        elif mood_config['aggression'] < 0.3:  # Saver mode
            if ml_action == 'rebalance' and ml_confidence > 0.7:
                return Action.REBALANCE, 0.8, f"ML predicts {ml_action} with {ml_confidence:.1%} confidence - risk management recommended"
            elif volatility > 50 or concentration > 0.7:
                return Action.REBALANCE, 0.8, "High risk detected - rebalancing recommended for safety"
            elif var_amount > wallet_data.get('total_value_usd', 10000) * 0.1:
                return Action.HOLD, 0.85, "High VaR - conservative approach recommended"
            else:
                return Action.HOLD, 0.7, "Stable conditions - maintain current positions"
        
        else:  # Balanced mode
            if ml_action in ['swap', 'rebalance'] and ml_confidence > 0.65:
                action_map = {'swap': Action.SWAP, 'rebalance': Action.REBALANCE}
                return action_map[ml_action], 0.7, f"ML predicts {ml_action} with {ml_confidence:.1%} confidence - balanced approach"
            elif volatility > 40 and concentration > 0.5:
                return Action.REBALANCE, 0.75, "Risk metrics suggest rebalancing for better diversification"
            elif fourier_pattern in ['daily', 'weekly'] and stat_features['avg_transaction_size'] > 1000:
                return Action.SWAP, 0.7, "Regular trading pattern detected with significant volume"
            else:
                return Action.HOLD, 0.65, "Balanced conditions - maintain current strategy"
    
    def _determine_action(self, wallet_data: Dict, risk_metrics: Dict, features: Dict, mood_config: Dict) -> tuple:
        """Determine the recommended action based on analysis (legacy method)"""
        
        # Extract key indicators
        volatility = risk_metrics.get('volatility_risk', {}).get('volatility_score', 50.0)
        var_amount = risk_metrics.get('var_risk', {}).get('var_95_dollar', 0)
        concentration = risk_metrics.get('concentration_risk', {}).get('hhi_score', 0)
        
        # Trading pattern analysis
        fourier_pattern = features.get('fourier_features', {}).get('pattern_type', 'unknown')
        stat_features = features.get('statistical_features', {})
        
        # Decision logic based on mood
        if mood_config['aggression'] > 0.7:  # Degen mode
            if volatility < 30 and concentration < 0.3:
                return Action.SWAP, 0.75, "Low volatility and good diversification - good time for aggressive trading"
            elif stat_features['whale_transaction_ratio'] > 0.5:
                return Action.ADD_LIQUIDITY, 0.65, "High whale activity detected - liquidity provision opportunity"
            else:
                return Action.HOLD, 0.6, "High risk environment - hold and wait for better opportunities"
        
        elif mood_config['aggression'] < 0.3:  # Saver mode
            if volatility > 50 or concentration > 0.7:
                return Action.REBALANCE, 0.8, "High risk detected - rebalancing recommended for safety"
            elif var_amount > wallet_data.get('total_value_usd', 10000) * 0.1:
                return Action.HOLD, 0.85, "High VaR - conservative approach recommended"
            else:
                return Action.HOLD, 0.7, "Stable conditions - maintain current positions"
        
        else:  # Balanced mode
            if volatility > 40 and concentration > 0.5:
                return Action.REBALANCE, 0.75, "Risk metrics suggest rebalancing for better diversification"
            elif fourier_pattern in ['daily', 'weekly'] and stat_features['avg_transaction_size'] > 1000:
                return Action.SWAP, 0.7, "Regular trading pattern detected with significant volume"
            else:
                return Action.HOLD, 0.65, "Balanced conditions - maintain current strategy"
    
    def _categorize_risk(self, risk_score: float) -> RiskCategory:
        """Categorize risk based on score"""
        if risk_score < 25:
            return RiskCategory.LOW
        elif risk_score < 50:
            return RiskCategory.MEDIUM
        elif risk_score < 75:
            return RiskCategory.HIGH
        else:
            return RiskCategory.VERY_HIGH
    
    def _get_fallback_recommendation(self) -> Dict[str, Any]:
        """Fallback recommendation when analysis fails"""
        return {
            'action': Action.HOLD,
            'confidence': 0.5,
            'reasoning': "Unable to complete full analysis - conservative hold recommendation",
            'risk_score': 50.0,
            'risk_category': RiskCategory.MEDIUM,
            'timestamp': datetime.now().isoformat(),
            'metadata': {'fallback': True}
        }
    
    def explain_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Generate human-readable explanation of recommendation"""
        action = recommendation['action']
        confidence = recommendation['confidence']
        reasoning = recommendation['reasoning']
        risk_score = recommendation['risk_score']
        
        # Safely get metadata values with defaults
        metadata = recommendation.get('metadata', {})
        ml_insights = recommendation.get('ml_insights', {})
        portfolio_value = metadata.get('portfolio_value', 0)
        volatility = metadata.get('volatility', 0)
        var_amount = metadata.get('var_amount', 0)
        concentration_risk = metadata.get('concentration_risk', 0)
        
        # Extract ML insights
        ml_action = ml_insights.get('predicted_action', 'hold')
        ml_confidence = ml_insights.get('action_confidence', 0.5)
        recommended_timing = ml_insights.get('recommended_timing', 'within_24h')
        recommended_size = ml_insights.get('recommended_size_percentage', 10.0)
        cluster_characteristics = ml_insights.get('cluster_characteristics', {})
        similar_wallets = ml_insights.get('similar_wallets', [])
        
        explanation = f"""
🎯 **Recommendation: {action.upper()}**
📊 **Confidence: {confidence:.1%}**
⚠️ **Risk Score: {risk_score:.1f}/100**

💡 **Reasoning:**
{reasoning}

🤖 **ML Insights:**
• **Predicted Action:** {ml_action.upper()} (Confidence: {ml_confidence:.1%})
• **Optimal Timing:** {recommended_timing.replace('_', ' ').title()}
• **Recommended Size:** {recommended_size:.1f}% of portfolio
• **Wallet Cluster:** {cluster_characteristics.get('trading_style', 'Unknown')} ({cluster_characteristics.get('risk_tolerance', 'Unknown')} risk)

🔍 **Key Factors:**
- Portfolio Value: ${portfolio_value:,.0f}
- Volatility: {volatility:.1f}%
- VaR: ${var_amount:,.0f}
- Concentration Risk: {concentration_risk:.2f}

⏰ **Generated: {recommendation.get('timestamp', 'Unknown')}**
        """
        
        return explanation.strip()
