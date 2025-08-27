"""
DefiLlama Risk Assessment Tool

This module provides comprehensive risk assessment capabilities using DefiLlama API endpoints.
It implements the critical risk assessment endpoints outlined in defillama.md to enrich
wallet analysis with protocol risk data, hack detection, and yield sustainability analysis.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from src.config import Config

logger = logging.getLogger(__name__)

@dataclass
class RiskThresholds:
    """Risk thresholds for different risk factors"""
    tvl_decline: float = -20  # 20% TVL drop in 7 days
    hack_recency: int = 90  # Hack within 90 days
    apy_unsustainable: float = 100  # APY over 100%
    concentration_high: float = 0.5  # 50%+ in single token
    oracle_tvs_low: float = 1000000  # <$1M secured
    stablecoin_depeg: float = 0.02  # 2% off peg
    liquidity_low: float = 100000  # <$100k daily volume

@dataclass
class RiskWeights:
    """Weights for different risk factors in overall risk calculation"""
    tvl_risk: float = 0.30
    hack_risk: float = 0.25
    concentration_risk: float = 0.15
    yield_risk: float = 0.15
    oracle_risk: float = 0.10
    bridge_risk: float = 0.05

class DefiLlamaRiskAnalyzer:
    """
    Comprehensive risk analyzer using DefiLlama API endpoints.
    
    Implements the critical risk assessment endpoints from defillama.md:
    - Protocol TVL health monitoring
    - Hack history and exploit detection
    - Yield sustainability analysis
    - Stablecoin depeg risk monitoring
    - Chain concentration risk assessment
    - Liquidity and slippage risk analysis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.defillama_base_url
        self.timeout = config.defillama_timeout
        self.session = requests.Session()
        
        # Set up headers if API key is provided
        if config.defillama_api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {config.defillama_api_key}',
                'Content-Type': 'application/json'
            })
        
        self.risk_thresholds = RiskThresholds()
        self.risk_weights = RiskWeights()
        
        logger.info("DefiLlamaRiskAnalyzer initialized")
    
    def _map_protocol_name(self, protocol_name: str) -> str:
        """
        Map protocol names to DefiLlama format.
        
        Args:
            protocol_name: Original protocol name
            
        Returns:
            Mapped protocol name for DefiLlama API
        """
        # Protocol name mapping for DefiLlama
        protocol_mapping = {
            'ethereum name service': 'ens',
            'across v2': 'across',
            'gaslite': 'gaslite',
            'circle cctp': 'circle',
            'cow swap': 'cowswap',
            'zerion': 'zerion',
            'anime': 'anime',
            'blur': 'blur',
            'metamask': 'metamask',
            'opensea': 'opensea',
            'uniswap': 'uniswap',
            'aave': 'aave',
            'compound': 'compound',
            'camelot': 'camelot',
            'symbiosis': 'symbiosis',
            'li.fi': 'lifi',
            'extra finance': 'extra-finance',
            'cbridge': 'cbridge'
        }
        
        # Clean the protocol name
        clean_name = protocol_name.lower().strip()
        
        # Return mapped name or original if not found
        return protocol_mapping.get(clean_name, clean_name)
    
    def get_protocol_tvl(self, protocol_name: str) -> Dict[str, Any]:
        """
        Get protocol TVL data and health metrics.
        
        Args:
            protocol_name: Name of the protocol (e.g., 'aave', 'uniswap')
            
        Returns:
            Dict containing TVL data, trends, and health metrics
        """
        try:
            # Map protocol name and URL encode it
            mapped_name = self._map_protocol_name(protocol_name)
            encoded_name = requests.utils.quote(mapped_name)
            url = f"{self.base_url}/protocol/{encoded_name}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Calculate TVL trends
            tvl_data = data.get('tvl', [])
            if len(tvl_data) >= 2:
                current_tvl = tvl_data[-1].get('totalLiquidityUSD', 0)
                week_ago_tvl = tvl_data[-8].get('totalLiquidityUSD', 0) if len(tvl_data) >= 8 else current_tvl
                month_ago_tvl = tvl_data[-31].get('totalLiquidityUSD', 0) if len(tvl_data) >= 31 else current_tvl
                
                week_change = ((current_tvl - week_ago_tvl) / week_ago_tvl * 100) if week_ago_tvl > 0 else 0
                month_change = ((current_tvl - month_ago_tvl) / month_ago_tvl * 100) if month_ago_tvl > 0 else 0
            else:
                current_tvl = 0
                week_change = 0
                month_change = 0
            
            return {
                'protocol': protocol_name,
                'current_tvl': current_tvl,
                'week_change_pct': week_change,
                'month_change_pct': month_change,
                'tvl_data': tvl_data,
                'chain_tvls': data.get('chainTvls', {}),
                'tokens_in_usd': data.get('tokensInUsd', []),
                'tokens': data.get('tokens', []),
                'last_updated': data.get('lastHourlyUpdate', 0)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch TVL data for {protocol_name}: {e}")
            return {
                'protocol': protocol_name,
                'error': str(e),
                'current_tvl': 0,
                'week_change_pct': 0,
                'month_change_pct': 0,
                'tvl_data': [],
                'chain_tvls': {},
                'tokens_in_usd': [],
                'tokens': [],
                'last_updated': 0
            }
    
    def get_hack_history(self, days_back: int = 365) -> List[Dict[str, Any]]:
        """
        Get hack history for risk assessment.
        
        Args:
            days_back: Number of days to look back for hacks
            
        Returns:
            List of hack events with details
        """
        try:
            url = f"{self.base_url}/hacks"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            hacks = response.json()
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            recent_hacks = [
                hack for hack in hacks 
                if hack.get('date', 0) >= cutoff_timestamp
            ]
            
            logger.info(f"Found {len(recent_hacks)} hacks in the last {days_back} days")
            return recent_hacks
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch hack history: {e}")
            return []
    
    def get_yield_pools(self, protocol_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get yield pool data for sustainability analysis.
        
        Args:
            protocol_name: Optional protocol filter
            
        Returns:
            List of yield pools with APY and risk metrics
        """
        try:
            # Use the correct endpoint for yields (v2)
            url = f"{self.base_url}/v2/yields"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            pools = response.json()
            
            # Filter by protocol if specified
            if protocol_name:
                pools = [pool for pool in pools if pool.get('protocol', '').lower() == protocol_name.lower()]
            
            # Add risk assessment
            for pool in pools:
                apy = pool.get('apy', 0)
                pool['risk_level'] = self._assess_yield_risk(apy)
                pool['sustainable'] = apy <= self.risk_thresholds.apy_unsustainable
            
            logger.info(f"Found {len(pools)} yield pools")
            return pools
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch yield pools: {e}")
            return []
    
    def get_stablecoin_data(self) -> Dict[str, Any]:
        """
        Get stablecoin market data for depeg risk assessment.
        
        Returns:
            Dict containing stablecoin market caps and risk metrics
        """
        try:
            url = f"{self.base_url}/stablecoins"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Calculate total market cap and identify risks
            total_mcap = sum(coin.get('marketCap', 0) for coin in data.get('peggedAssets', []))
            
            # Identify potential depeg risks
            depeg_risks = []
            for coin in data.get('peggedAssets', []):
                if coin.get('priceSource') == 'chainlink':
                    price = coin.get('price', 1.0)
                    if abs(price - 1.0) > self.risk_thresholds.stablecoin_depeg:
                        depeg_risks.append({
                            'symbol': coin.get('symbol', ''),
                            'price': price,
                            'deviation': abs(price - 1.0) * 100
                        })
            
            return {
                'total_market_cap': total_mcap,
                'pegged_assets': data.get('peggedAssets', []),
                'depeg_risks': depeg_risks,
                'risk_count': len(depeg_risks)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch stablecoin data: {e}")
            return {'error': str(e)}
    
    def get_chain_tvls(self) -> List[Dict[str, Any]]:
        """
        Get chain TVL data for concentration risk assessment.
        
        Returns:
            List of chains with TVL and risk metrics
        """
        try:
            url = f"{self.base_url}/v2/chains"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            chains = response.json()
            
            # Calculate total TVL
            total_tvl = sum(chain.get('tvl', 0) for chain in chains)
            
            # Add concentration metrics
            for chain in chains:
                tvl = chain.get('tvl', 0)
                chain['concentration_pct'] = (tvl / total_tvl * 100) if total_tvl > 0 else 0
                chain['risk_level'] = self._assess_chain_concentration_risk(chain['concentration_pct'])
            
            return chains
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch chain TVLs: {e}")
            return []
    
    def get_dex_volume(self, protocol_name: str) -> Dict[str, Any]:
        """
        Get DEX volume data for liquidity risk assessment.
        
        Args:
            protocol_name: Name of the DEX protocol
            
        Returns:
            Dict containing volume data and liquidity metrics
        """
        try:
            url = f"{self.base_url}/dexs/{protocol_name}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Calculate average daily volume
            protocols = data.get('protocols', [])
            total_volume_24h = sum(protocol.get('volume24h', 0) for protocol in protocols)
            
            return {
                'protocol': protocol_name,
                'total_volume_24h': total_volume_24h,
                'protocols': protocols,
                'liquidity_risk': self._assess_liquidity_risk(total_volume_24h)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch DEX volume for {protocol_name}: {e}")
            return {
                'protocol': protocol_name,
                'error': str(e),
                'total_volume_24h': 0,
                'liquidity_risk': 'high'
            }
    
    def calculate_protocol_risk_score(self, protocol_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a protocol.
        
        Args:
            protocol_name: Name of the protocol to analyze
            
        Returns:
            Dict containing risk score, factors, and recommendations
        """
        risk_factors = {}
        
        # 1. TVL Trend Risk (30% weight)
        tvl_data = self.get_protocol_tvl(protocol_name)
        risk_factors['tvl_risk'] = self._calculate_tvl_risk(tvl_data)
        
        # 2. Hack History Risk (25% weight)
        hack_data = self.get_hack_history()
        risk_factors['hack_risk'] = self._calculate_hack_risk(protocol_name, hack_data)
        
        # 3. Concentration Risk (15% weight)
        risk_factors['concentration_risk'] = self._calculate_concentration_risk(tvl_data)
        
        # 4. Yield Sustainability Risk (15% weight)
        yield_data = self.get_yield_pools(protocol_name)
        risk_factors['yield_risk'] = self._calculate_yield_risk(yield_data)
        
        # 5. Oracle Risk (10% weight) - Simplified for now
        risk_factors['oracle_risk'] = 0.5  # Default medium risk
        
        # 6. Bridge Exposure Risk (5% weight) - Simplified for now
        risk_factors['bridge_risk'] = 0.3  # Default low risk
        
        # Calculate weighted average
        total_risk = sum(
            risk_factors[key] * getattr(self.risk_weights, key)
            for key in risk_factors
        )
        
        return {
            'protocol': protocol_name,
            'total_risk_score': total_risk,
            'risk_factors': risk_factors,
            'risk_level': self._categorize_risk(total_risk),
            'recommendations': self._generate_risk_recommendations(risk_factors),
            'tvl_data': tvl_data,
            'yield_data': yield_data
        }
    
    def analyze_wallet_protocols(self, wallet_protocols: List[str]) -> Dict[str, Any]:
        """
        Analyze risk for all protocols used by a wallet.
        
        Args:
            wallet_protocols: List of protocol names used by the wallet
            
        Returns:
            Dict containing overall risk assessment and protocol-specific risks
        """
        protocol_risks = {}
        total_risk_score = 0
        
        # Limit to top 10 protocols to avoid API rate limits
        limited_protocols = wallet_protocols[:10]
        
        for protocol in limited_protocols:
            try:
                risk_data = self.calculate_protocol_risk_score(protocol)
                protocol_risks[protocol] = risk_data
                total_risk_score += risk_data['total_risk_score']
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to analyze protocol {protocol}: {e}")
                protocol_risks[protocol] = {
                    'error': str(e),
                    'total_risk_score': 0.5,  # Default medium risk
                    'risk_level': 'medium'
                }
        
        # Calculate average risk
        avg_risk = total_risk_score / len(wallet_protocols) if wallet_protocols else 0
        
        return {
            'wallet_protocol_risk': avg_risk,
            'risk_level': self._categorize_risk(avg_risk),
            'protocol_risks': protocol_risks,
            'high_risk_protocols': [
                p for p, data in protocol_risks.items()
                if data.get('risk_level') == 'high'
            ],
            'recommendations': self._generate_wallet_recommendations(protocol_risks)
        }
    
    # Private helper methods
    
    def _calculate_tvl_risk(self, tvl_data: Dict[str, Any]) -> float:
        """Calculate TVL trend risk score (0-1)"""
        if 'error' in tvl_data:
            return 0.8  # High risk if we can't get data
        
        week_change = tvl_data.get('week_change_pct', 0)
        month_change = tvl_data.get('month_change_pct', 0)
        
        # Higher risk for declining TVL
        if week_change < self.risk_thresholds.tvl_decline:
            return 0.9
        elif week_change < -10:
            return 0.7
        elif week_change < 0:
            return 0.5
        else:
            return 0.2
    
    def _calculate_hack_risk(self, protocol_name: str, hack_data: List[Dict[str, Any]]) -> float:
        """Calculate hack risk score (0-1)"""
        # Check for direct hacks on this protocol
        direct_hacks = [
            hack for hack in hack_data
            if hack.get('name', '').lower() == protocol_name.lower()
        ]
        
        if direct_hacks:
            # Check if hack was recent
            latest_hack = max(direct_hacks, key=lambda x: x.get('date', 0))
            days_since_hack = (datetime.now() - datetime.fromtimestamp(latest_hack.get('date', 0))).days
            
            if days_since_hack <= self.risk_thresholds.hack_recency:
                return 0.9  # Very high risk for recent hacks
            else:
                return 0.6  # Medium risk for older hacks
        
        # Check for similar protocol hacks (same classification)
        protocol_classifications = set()
        for hack in hack_data:
            if hack.get('name', '').lower() == protocol_name.lower():
                protocol_classifications.add(hack.get('classification', ''))
        
        if protocol_classifications:
            return 0.4  # Medium risk if similar protocols were hacked
        
        return 0.2  # Low risk if no hacks found
    
    def _calculate_concentration_risk(self, tvl_data: Dict[str, Any]) -> float:
        """Calculate concentration risk score (0-1)"""
        if 'error' in tvl_data:
            return 0.5  # Medium risk if we can't get data
        
        tokens = tvl_data.get('tokens', [])
        if not tokens:
            return 0.3
        
        # Calculate concentration using top token
        total_value = sum(token.get('tvl', 0) for token in tokens)
        if total_value == 0:
            return 0.3
        
        top_token_value = max(token.get('tvl', 0) for token in tokens)
        concentration = top_token_value / total_value
        
        if concentration > self.risk_thresholds.concentration_high:
            return 0.8
        elif concentration > 0.3:
            return 0.5
        else:
            return 0.2
    
    def _calculate_yield_risk(self, yield_data: List[Dict[str, Any]]) -> float:
        """Calculate yield sustainability risk score (0-1)"""
        if not yield_data:
            return 0.3  # Low risk if no yield data
        
        high_risk_pools = [
            pool for pool in yield_data
            if pool.get('apy', 0) > self.risk_thresholds.apy_unsustainable
        ]
        
        risk_ratio = len(high_risk_pools) / len(yield_data)
        
        if risk_ratio > 0.5:
            return 0.9
        elif risk_ratio > 0.2:
            return 0.6
        else:
            return 0.3
    
    def _assess_yield_risk(self, apy: float) -> str:
        """Assess yield risk level"""
        if apy > self.risk_thresholds.apy_unsustainable:
            return 'very_high'
        elif apy > 50:
            return 'high'
        elif apy > 20:
            return 'medium'
        else:
            return 'low'
    
    def _assess_chain_concentration_risk(self, concentration_pct: float) -> str:
        """Assess chain concentration risk"""
        if concentration_pct > 50:
            return 'very_high'
        elif concentration_pct > 30:
            return 'high'
        elif concentration_pct > 15:
            return 'medium'
        else:
            return 'low'
    
    def _assess_liquidity_risk(self, volume_24h: float) -> str:
        """Assess liquidity risk based on 24h volume"""
        if volume_24h < self.risk_thresholds.liquidity_low:
            return 'high'
        elif volume_24h < 1000000:  # $1M
            return 'medium'
        else:
            return 'low'
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into level"""
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_recommendations(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk-specific recommendations"""
        recommendations = []
        
        if risk_factors.get('tvl_risk', 0) > 0.6:
            recommendations.append("Consider reducing exposure due to declining TVL")
        
        if risk_factors.get('hack_risk', 0) > 0.6:
            recommendations.append("High hack risk - consider migrating to safer protocols")
        
        if risk_factors.get('yield_risk', 0) > 0.6:
            recommendations.append("Unsustainable yields detected - exit high-APY positions")
        
        if risk_factors.get('concentration_risk', 0) > 0.6:
            recommendations.append("High concentration risk - diversify across more tokens")
        
        if not recommendations:
            recommendations.append("Protocol appears relatively safe")
        
        return recommendations
    
    def _generate_wallet_recommendations(self, protocol_risks: Dict[str, Any]) -> List[str]:
        """Generate wallet-level recommendations"""
        recommendations = []
        
        high_risk_count = len([
            p for p, data in protocol_risks.items()
            if data.get('risk_level') == 'high'
        ])
        
        if high_risk_count > 0:
            recommendations.append(f"Consider exiting {high_risk_count} high-risk protocols")
        
        if len(protocol_risks) < 3:
            recommendations.append("Diversify across more protocols to reduce concentration risk")
        
        return recommendations
