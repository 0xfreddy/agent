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
            
            # Step 4: Compile Analysis Results
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
            
            # Calculate risk-adjusted metrics
            base_risk_score = risk_metrics['volatility']['volatility_score']
            adjusted_risk_score = base_risk_score * mood_config['risk_multiplier']
            
            # Determine action based on analysis
            action, confidence, reasoning = self._determine_action(
                wallet_data, risk_metrics, features, mood_config
            )
            
            # Generate recommendation
            recommendation = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'risk_score': adjusted_risk_score,
                'risk_category': self._categorize_risk(adjusted_risk_score),
                'mood_adjustment': mood_config,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'portfolio_value': wallet_data['total_value_usd'],
                    'transaction_count': wallet_data['transaction_count'],
                    'volatility': risk_metrics['volatility']['volatility_percentage'],
                    'var_amount': risk_metrics['var']['var_95_dollar'],
                    'concentration_risk': risk_metrics['concentration']['hhi_score']
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
    
    def _determine_action(self, wallet_data: Dict, risk_metrics: Dict, features: Dict, mood_config: Dict) -> tuple:
        """Determine the recommended action based on analysis"""
        
        # Extract key indicators
        volatility = risk_metrics['volatility']['volatility_score']
        var_amount = risk_metrics['var']['var_95_dollar']
        concentration = risk_metrics['concentration']['hhi_score']
        
        # Trading pattern analysis
        fourier_pattern = features['fourier']['pattern_type']
        stat_features = features['statistical']
        
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
            elif var_amount > wallet_data['total_value_usd'] * 0.1:
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
        portfolio_value = metadata.get('portfolio_value', 0)
        volatility = metadata.get('volatility', 0)
        var_amount = metadata.get('var_amount', 0)
        concentration_risk = metadata.get('concentration_risk', 0)
        
        explanation = f"""
🎯 **Recommendation: {action.upper()}**
📊 **Confidence: {confidence:.1%}**
⚠️ **Risk Score: {risk_score:.1f}/100**

💡 **Reasoning:**
{reasoning}

🔍 **Key Factors:**
- Portfolio Value: ${portfolio_value:,.0f}
- Volatility: {volatility:.1f}%
- VaR: ${var_amount:,.0f}
- Concentration Risk: {concentration_risk:.2f}

⏰ **Generated: {recommendation.get('timestamp', 'Unknown')}**
        """
        
        return explanation.strip()
