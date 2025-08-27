import numpy as np
from scipy import stats
from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

class PortfolioVolatilityTool(BaseTool):
    name = "portfolio_volatility"
    description = "Calculates portfolio volatility using covariance matrix"
    
    def _run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            balances = portfolio_data.get('balances', {})
            price_history = portfolio_data.get('price_history', {})
            
            if not balances or len(balances) < 2:
                return {
                    'volatility_score': 0.0,
                    'volatility_percentage': 0.0,
                    'risk_level': 'low',
                    'diversification_ratio': 1.0
                }
            
            weights = np.array([b.get('allocation_percentage', 0) / 100 for b in balances.values()])
            
            returns = self._calculate_returns(price_history)
            
            if returns.size == 0:
                returns = self._generate_mock_returns(len(balances))
            
            cov_matrix = np.cov(returns.T) if returns.shape[0] > 1 else np.array([[0.01]])
            
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            annualized_volatility = portfolio_volatility * np.sqrt(365)
            
            volatility_score = min(100, annualized_volatility * 100)
            
            risk_level = self._determine_risk_level(volatility_score)
            
            individual_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_volatility = np.dot(weights, individual_volatilities)
            diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            return {
                'volatility_score': float(volatility_score),
                'volatility_percentage': float(annualized_volatility * 100),
                'risk_level': risk_level,
                'diversification_ratio': float(diversification_ratio),
                'correlation_benefit': float(max(0, diversification_ratio - 1))
            }
            
        except Exception as e:
            logging.error(f"Portfolio volatility calculation error: {e}")
            return {
                'volatility_score': 50.0,
                'volatility_percentage': 25.0,
                'risk_level': 'medium',
                'diversification_ratio': 1.0
            }
    
    def _calculate_returns(self, price_history: Dict) -> np.ndarray:
        if not price_history:
            return np.array([])
        
        returns_list = []
        for token, prices in price_history.items():
            if len(prices) > 1:
                prices_array = np.array(prices)
                returns = np.diff(prices_array) / prices_array[:-1]
                returns_list.append(returns)
        
        if not returns_list:
            return np.array([])
        
        min_length = min(len(r) for r in returns_list)
        aligned_returns = np.array([r[:min_length] for r in returns_list])
        
        return aligned_returns.T
    
    def _generate_mock_returns(self, n_assets: int) -> np.ndarray:
        n_periods = 30
        mean_returns = np.random.uniform(-0.01, 0.03, n_assets)
        std_devs = np.random.uniform(0.01, 0.05, n_assets)
        
        returns = np.random.normal(mean_returns, std_devs, (n_periods, n_assets))
        return returns
    
    def _determine_risk_level(self, volatility_score: float) -> str:
        if volatility_score < 20:
            return 'low'
        elif volatility_score < 40:
            return 'medium'
        elif volatility_score < 60:
            return 'high'
        else:
            return 'very_high'

class VaRCalculatorTool(BaseTool):
    name = "var_calculator"
    description = "Calculate Value at Risk at 95% confidence"
    
    def _run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            total_value = portfolio_data.get('total_value_usd', 0)
            returns_history = portfolio_data.get('returns_history', [])
            
            if not total_value or not returns_history:
                return self._calculate_parametric_var(portfolio_data)
            
            returns_array = np.array(returns_history)
            
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            var_95_dollar = abs(var_95 * total_value)
            var_99_dollar = abs(var_99 * total_value)
            
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
            cvar_95_dollar = abs(cvar_95 * total_value)
            
            return {
                'var_95_percentage': float(abs(var_95) * 100),
                'var_95_dollar': float(var_95_dollar),
                'var_99_percentage': float(abs(var_99) * 100),
                'var_99_dollar': float(var_99_dollar),
                'cvar_95_dollar': float(cvar_95_dollar),
                'risk_assessment': self._assess_var_risk(var_95),
                'max_expected_loss_1day': float(var_95_dollar),
                'max_expected_loss_1week': float(var_95_dollar * np.sqrt(7))
            }
            
        except Exception as e:
            logging.error(f"VaR calculation error: {e}")
            return self._get_default_var()
    
    def _calculate_parametric_var(self, portfolio_data: Dict) -> Dict[str, Any]:
        total_value = portfolio_data.get('total_value_usd', 10000)
        
        assumed_mean = 0.001
        assumed_std = 0.02
        
        z_score_95 = 1.645
        z_score_99 = 2.326
        
        var_95 = assumed_mean - (assumed_std * z_score_95)
        var_99 = assumed_mean - (assumed_std * z_score_99)
        
        return {
            'var_95_percentage': float(abs(var_95) * 100),
            'var_95_dollar': float(abs(var_95 * total_value)),
            'var_99_percentage': float(abs(var_99) * 100),
            'var_99_dollar': float(abs(var_99 * total_value)),
            'cvar_95_dollar': float(abs(var_95 * total_value * 1.2)),
            'risk_assessment': 'medium',
            'max_expected_loss_1day': float(abs(var_95 * total_value)),
            'max_expected_loss_1week': float(abs(var_95 * total_value * np.sqrt(7)))
        }
    
    def _assess_var_risk(self, var_95: float) -> str:
        var_abs = abs(var_95)
        if var_abs < 0.02:
            return 'low'
        elif var_abs < 0.05:
            return 'medium'
        elif var_abs < 0.10:
            return 'high'
        else:
            return 'very_high'
    
    def _get_default_var(self) -> Dict[str, Any]:
        return {
            'var_95_percentage': 3.29,
            'var_95_dollar': 329.0,
            'var_99_percentage': 4.65,
            'var_99_dollar': 465.0,
            'cvar_95_dollar': 400.0,
            'risk_assessment': 'medium',
            'max_expected_loss_1day': 329.0,
            'max_expected_loss_1week': 870.0
        }

class ConcentrationRiskTool(BaseTool):
    name = "concentration_risk"
    description = "Calculate Herfindahl-Hirschman Index for concentration risk"
    
    def _run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            balances = portfolio_data.get('balances', {})
            
            if not balances:
                return {
                    'hhi_score': 10000,
                    'concentration_risk': 1.0,
                    'risk_level': 'very_high',
                    'diversification_score': 0.0,
                    'top_3_concentration': 100.0
                }
            
            allocations = [b.get('allocation_percentage', 0) for b in balances.values()]
            
            if not allocations or sum(allocations) == 0:
                return {
                    'hhi_score': 10000,
                    'concentration_risk': 1.0,
                    'risk_level': 'very_high',
                    'diversification_score': 0.0,
                    'top_3_concentration': 100.0
                }
            
            normalized_allocations = [a / sum(allocations) * 100 for a in allocations]
            
            hhi = sum([(s/100) ** 2 for s in normalized_allocations]) * 10000
            
            concentration_risk = hhi / 10000
            
            risk_level = self._determine_concentration_risk_level(hhi)
            
            n_assets = len(allocations)
            min_hhi = 10000 / n_assets if n_assets > 0 else 10000
            diversification_score = max(0, (1 - (hhi - min_hhi) / (10000 - min_hhi))) * 100 if n_assets > 1 else 0
            
            sorted_allocations = sorted(normalized_allocations, reverse=True)
            top_3_concentration = sum(sorted_allocations[:3]) if len(sorted_allocations) >= 3 else sum(sorted_allocations)
            
            return {
                'hhi_score': float(hhi),
                'concentration_risk': float(concentration_risk),
                'risk_level': risk_level,
                'diversification_score': float(diversification_score),
                'top_3_concentration': float(top_3_concentration),
                'number_of_positions': n_assets,
                'effective_number': float(10000 / hhi) if hhi > 0 else 1,
                'recommendation': self._get_concentration_recommendation(hhi, n_assets)
            }
            
        except Exception as e:
            logging.error(f"Concentration risk calculation error: {e}")
            return {
                'hhi_score': 5000,
                'concentration_risk': 0.5,
                'risk_level': 'medium',
                'diversification_score': 50.0,
                'top_3_concentration': 75.0
            }
    
    def _determine_concentration_risk_level(self, hhi: float) -> str:
        if hhi < 1500:
            return 'low'
        elif hhi < 2500:
            return 'medium'
        elif hhi < 5000:
            return 'high'
        else:
            return 'very_high'
    
    def _get_concentration_recommendation(self, hhi: float, n_assets: int) -> str:
        if hhi > 5000:
            return "High concentration risk. Consider diversifying into more assets."
        elif hhi > 2500:
            return "Moderate concentration. Could benefit from additional diversification."
        elif n_assets < 3:
            return "Well-balanced but limited positions. Consider adding 1-2 more assets."
        else:
            return "Good diversification level."

class ProtocolRiskTool(BaseTool):
    name = "protocol_risk"
    description = "Assess protocol risk based on multiple factors"
    
    def _run(self, protocol_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            protocol_name = protocol_data.get('protocol_name', 'unknown')
            tvl = protocol_data.get('tvl', 0)
            tvl_change_30d = protocol_data.get('tvl_change_30d', 0)
            audit_status = protocol_data.get('audit_status', False)
            time_since_launch_days = protocol_data.get('time_since_launch_days', 0)
            recent_exploits = protocol_data.get('recent_exploits', [])
            
            risk_score = 50.0
            risk_factors = {}
            
            if tvl > 1_000_000_000:
                risk_score -= 15
                risk_factors['tvl'] = 'High TVL (low risk)'
            elif tvl > 100_000_000:
                risk_score -= 5
                risk_factors['tvl'] = 'Medium TVL (moderate risk)'
            elif tvl < 10_000_000:
                risk_score += 20
                risk_factors['tvl'] = 'Low TVL (high risk)'
            else:
                risk_factors['tvl'] = 'Standard TVL'
            
            if tvl_change_30d < -50:
                risk_score += 15
                risk_factors['tvl_trend'] = 'Rapid TVL decline (high risk)'
            elif tvl_change_30d < -20:
                risk_score += 5
                risk_factors['tvl_trend'] = 'TVL declining (moderate risk)'
            elif tvl_change_30d > 100:
                risk_score += 10
                risk_factors['tvl_trend'] = 'Rapid TVL growth (potential risk)'
            else:
                risk_factors['tvl_trend'] = 'Stable TVL'
            
            if audit_status:
                risk_score -= 10
                risk_factors['audit'] = 'Audited (low risk)'
            else:
                risk_score += 15
                risk_factors['audit'] = 'No audit (high risk)'
            
            if time_since_launch_days > 365:
                risk_score -= 10
                risk_factors['maturity'] = 'Mature protocol (low risk)'
            elif time_since_launch_days > 90:
                risk_factors['maturity'] = 'Established protocol'
            elif time_since_launch_days < 30:
                risk_score += 25
                risk_factors['maturity'] = 'New protocol (high risk)'
            else:
                risk_score += 10
                risk_factors['maturity'] = 'Young protocol (moderate risk)'
            
            if recent_exploits:
                risk_score += len(recent_exploits) * 15
                risk_factors['exploits'] = f'{len(recent_exploits)} recent exploits (very high risk)'
            else:
                risk_factors['exploits'] = 'No recent exploits'
            
            risk_score = max(0, min(100, risk_score))
            
            risk_category = self._determine_risk_category(risk_score)
            
            return {
                'protocol_name': protocol_name,
                'risk_score': float(risk_score),
                'risk_category': risk_category,
                'risk_factors': risk_factors,
                'tvl_millions': float(tvl / 1_000_000) if tvl else 0,
                'days_since_launch': time_since_launch_days,
                'is_audited': audit_status,
                'exploit_count': len(recent_exploits),
                'recommendation': self._get_protocol_recommendation(risk_score)
            }
            
        except Exception as e:
            logging.error(f"Protocol risk assessment error: {e}")
            return {
                'protocol_name': 'unknown',
                'risk_score': 50.0,
                'risk_category': 'medium',
                'risk_factors': {},
                'recommendation': 'Unable to assess risk'
            }
    
    def _determine_risk_category(self, risk_score: float) -> str:
        if risk_score < 25:
            return 'low'
        elif risk_score < 50:
            return 'medium'
        elif risk_score < 75:
            return 'high'
        else:
            return 'very_high'
    
    def _get_protocol_recommendation(self, risk_score: float) -> str:
        if risk_score < 25:
            return "Low risk protocol suitable for conservative strategies"
        elif risk_score < 50:
            return "Moderate risk - suitable for balanced portfolios"
        elif risk_score < 75:
            return "High risk - only for experienced users with high risk tolerance"
        else:
            return "Very high risk - exercise extreme caution or avoid"