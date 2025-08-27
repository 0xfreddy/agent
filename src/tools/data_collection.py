from langchain.tools import BaseTool
from typing import Dict, Any, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import redis
import json
import hashlib
from datetime import datetime, timedelta
import logging
from src.config import get_config
from src.models.schemas import WalletData, TransactionHistory, TokenBalance, MarketData, ProtocolData

class BaseDataTool:
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_enabled = True
        except:
            self.logger.warning("Redis not available, caching disabled")
            self.cache_enabled = False
            self.redis_client = None
    
    def _cache_key(self, prefix: str, params: Dict) -> str:
        params_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(params_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        if not self.cache_enabled:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                self.logger.info(f"Cache hit for {key}")
                return json.loads(cached)
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def _set_cached(self, key: str, data: Dict, ttl: int = None):
        if not self.cache_enabled:
            return
        
        ttl = ttl or self.config.cache_ttl
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(data)
            )
            self.logger.info(f"Cached {key} for {ttl} seconds")
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")

class OctavDataTool(BaseDataTool):
    name = "octav_wallet_data"
    description = "Fetches comprehensive wallet data from Octav API"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, wallet_address: str) -> Dict[str, Any]:
        cache_key = self._cache_key("octav", {"wallet": wallet_address})
        cached_data = self._get_cached(cache_key)
        
        if cached_data:
            return cached_data
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            # Get portfolio data from proxy server (for token balances)
            self.logger.info(f"Fetching portfolio data for {wallet_address}")
            portfolio_params = {
                "addresses": wallet_address,
                "includeNFTs": str(self.config.octav_include_nfts).lower(),
                "includeImages": str(self.config.octav_include_images).lower(),
                "includeExplorerUrls": str(self.config.octav_include_explorer_urls).lower(),
                "waitForSync": str(self.config.octav_wait_for_sync).lower()
            }
            portfolio_response = requests.get(
                f"http://localhost:3001/api/portfolio",
                headers=headers,
                params=portfolio_params,
                timeout=self.config.octav_portfolio_timeout
            )
            portfolio_response.raise_for_status()
            portfolio_data = portfolio_response.json()
            
            # Get wallet data from proxy server (for transaction history)
            self.logger.info(f"Fetching wallet data for {wallet_address}")
            wallet_response = requests.get(
                f"http://localhost:3001/api/wallet",
                headers=headers,
                params={"addresses": wallet_address},
                timeout=self.config.request_timeout
            )
            wallet_response.raise_for_status()
            wallet_data = wallet_response.json()
            
            # Parse the Octav API response structure
            transactions = self._parse_octav_transactions(wallet_data)
            balances = self._parse_octav_portfolio(portfolio_data)
            
            # Extract portfolio metrics if available
            portfolio_metrics = balances.pop('_portfolio_metrics', {})
            
            wallet_data = {
                'address': wallet_address.lower(),
                'transactions': transactions,
                'balances': balances,
                'total_value_usd': sum(b['value_usd'] for b in balances.values()),
                'transaction_count': len(transactions),
                'first_transaction': transactions[0]['timestamp'] if transactions else None,
                'last_transaction': transactions[-1]['timestamp'] if transactions else None,
                # Add comprehensive portfolio metrics
                'networth': portfolio_metrics.get('networth', 0),
                'cash_balance': portfolio_metrics.get('cash_balance', 0),
                'open_pnl': portfolio_metrics.get('open_pnl', 0),
                'closed_pnl': portfolio_metrics.get('closed_pnl', 0),
                'daily_income': portfolio_metrics.get('daily_income', 0),
                'daily_expense': portfolio_metrics.get('daily_expense', 0),
                'fees_fiat': portfolio_metrics.get('fees_fiat', 0),
                'last_updated': portfolio_metrics.get('last_updated', 'N/A')
            }
            
            self._set_cached(cache_key, wallet_data)
            
            return wallet_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Octav API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
                
                # Handle specific HTTP status codes
                if e.response.status_code == 401:
                    raise ValueError("Invalid API key. Please check your OCTAV_API_KEY configuration.")
                elif e.response.status_code == 403:
                    raise ValueError("API key does not have permission to access this endpoint.")
                elif e.response.status_code == 429:
                    raise ValueError("Rate limit exceeded. Please wait before making another request.")
                elif e.response.status_code == 500:
                    raise ValueError("Octav API server error. Please try again later.")
                elif e.response.status_code >= 400:
                    raise ValueError(f"API request failed with status {e.response.status_code}: {e.response.text}")
            
            # Handle network errors
            if isinstance(e, requests.exceptions.Timeout):
                raise ValueError("Request timed out. Please check your internet connection and try again.")
            elif isinstance(e, requests.exceptions.ConnectionError):
                raise ValueError("Connection error. Please check your internet connection and try again.")
            
            # If API fails completely, raise the error instead of returning mock data
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {e}")
            raise ValueError("Invalid JSON response from Octav API. Please try again later.")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise ValueError(f"An unexpected error occurred: {str(e)}")
    
    def _parse_octav_transactions(self, wallet_data: Dict) -> List[Dict]:
        """Parse transaction data from Octav /v1/wallet response"""
        transactions = []
        
        try:
            # Octav returns data in a specific structure - need to handle it properly
            if isinstance(wallet_data, dict):
                # Check for transactions in the response
                tx_list = wallet_data.get('transactions', [])
                if not tx_list and 'data' in wallet_data:
                    tx_list = wallet_data['data'].get('transactions', [])
                
                for tx in tx_list[:100]:
                    try:
                        transactions.append({
                            'timestamp': datetime.fromisoformat(tx.get('timestamp', '')) if tx.get('timestamp') else datetime.now(),
                            'protocol': tx.get('protocol', 'unknown'),
                            'asset_in': tx.get('tokenIn', tx.get('asset_in')),
                            'asset_out': tx.get('tokenOut', tx.get('asset_out')),
                            'value_usd': float(tx.get('valueUSD', tx.get('value_usd', 0))),
                            'gas_paid': float(tx.get('gasUsed', tx.get('gas_paid', 0))),
                            'tx_hash': tx.get('hash', tx.get('tx_hash', '0x' + '0' * 64)),
                            'method': tx.get('method', tx.get('type'))
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to parse transaction: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing Octav transactions: {e}")
        
        return transactions
    
    def _parse_octav_portfolio(self, portfolio_data: Dict) -> Dict[str, Dict]:
        """Parse portfolio data from Octav /v1/portfolio response with full hierarchical structure"""
        balances = {}
        
        try:
            # Handle Octav's response structure - it's a list of portfolio objects
            if isinstance(portfolio_data, list) and len(portfolio_data) > 0:
                portfolio = portfolio_data[0]  # Take first portfolio
            elif isinstance(portfolio_data, dict):
                portfolio = portfolio_data
            else:
                self.logger.warning("Unexpected portfolio data structure")
                return balances
            
            # Extract portfolio-level metrics with safe conversion
            def safe_float(value, default=0.0):
                if value == 'N/A' or value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            portfolio_metrics = {
                'networth': safe_float(portfolio.get('networth', 0)),
                'cash_balance': safe_float(portfolio.get('cashBalance', 0)),
                'open_pnl': safe_float(portfolio.get('openPnl', 0)),
                'closed_pnl': safe_float(portfolio.get('closedPnl', 0)),
                'daily_income': safe_float(portfolio.get('dailyIncome', 0)),
                'daily_expense': safe_float(portfolio.get('dailyExpense', 0)),
                'fees_fiat': safe_float(portfolio.get('feesFiat', 0)),
                'last_updated': portfolio.get('lastUpdated', 'N/A')
            }
            
            # Parse hierarchical structure: assetByProtocols -> chains -> protocolPositions -> assets
            asset_by_protocols = portfolio.get('assetByProtocols', {})
            
            for protocol_key, protocol_data in asset_by_protocols.items():
                protocol_name = protocol_data.get('name', protocol_key)
                protocol_value = float(protocol_data.get('value', 0))
                
                # Parse chains within each protocol
                chains = protocol_data.get('chains', {})
                for chain_key, chain_data in chains.items():
                    chain_name = chain_data.get('name', chain_key)
                    chain_value = float(chain_data.get('value', 0))
                    
                    # Parse protocol positions within each chain
                    protocol_positions = chain_data.get('protocolPositions', {})
                    for position_key, position_data in protocol_positions.items():
                        position_name = position_data.get('name', position_key)
                        position_value = float(position_data.get('totalValue', 0))
                        
                        # Parse individual assets within each position
                        assets = position_data.get('assets', [])
                        for asset in assets:
                            try:
                                symbol = asset.get('symbol', asset.get('name', 'UNKNOWN'))
                                balance_val = float(asset.get('balance', 0))
                                price = float(asset.get('price', 0))
                                value_usd = float(asset.get('value', balance_val * price))
                                
                                if balance_val > 0:  # Only include non-zero balances
                                    # Create unique key to avoid conflicts
                                    asset_key = f"{symbol}_{protocol_name}_{chain_name}"
                                    
                                    balances[asset_key] = {
                                        'symbol': symbol,
                                        'name': asset.get('name', symbol),
                                        'address': asset.get('address', ''),
                                        'balance': balance_val,
                                        'value_usd': value_usd,
                                        'price_usd': price,
                                        'allocation_percentage': 0,
                                        'protocol': protocol_name,
                                        'chain': chain_name,
                                        'position': position_name,
                                        'protocol_value': protocol_value,
                                        'chain_value': chain_value,
                                        'position_value': position_value
                                    }
                            except Exception as e:
                                self.logger.warning(f"Failed to parse asset data: {e}")
            
            # If no hierarchical data found, fall back to flat structure
            if not balances:
                self.logger.info("No hierarchical data found, trying flat structure")
                balances = self._parse_flat_portfolio_structure(portfolio)
            
            # Calculate allocation percentages
            total_value = sum(b['value_usd'] for b in balances.values())
            if total_value > 0:
                for token in balances:
                    balances[token]['allocation_percentage'] = (
                        balances[token]['value_usd'] / total_value * 100
                    )
            
            # Store portfolio metrics in a special key
            balances['_portfolio_metrics'] = portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Error parsing Octav portfolio: {e}")
            # Fall back to flat structure
            balances = self._parse_flat_portfolio_structure(portfolio_data)
        
        return balances
    
    def _parse_flat_portfolio_structure(self, portfolio_data: Dict) -> Dict[str, Dict]:
        """Fallback method to parse flat portfolio structure"""
        balances = {}
        
        try:
            if isinstance(portfolio_data, dict):
                # Check for tokens in various possible locations
                tokens = portfolio_data.get('tokens', portfolio_data.get('assets', []))
                
                # If it's still a dict, it might be the direct token list
                if isinstance(tokens, dict):
                    tokens = [{'symbol': k, **v} for k, v in tokens.items()]
                
                # Parse each token
                for token_data in tokens:
                    try:
                        symbol = token_data.get('symbol', token_data.get('ticker', 'UNKNOWN'))
                        balance_val = float(token_data.get('balance', token_data.get('amount', 0)))
                        price = float(token_data.get('price', token_data.get('priceUSD', 0)))
                        value_usd = float(token_data.get('valueUSD', balance_val * price))
                        
                        if balance_val > 0:  # Only include non-zero balances
                            balances[symbol] = {
                                'symbol': symbol,
                                'address': token_data.get('address', token_data.get('contractAddress', '')),
                                'balance': balance_val,
                                'value_usd': value_usd,
                                'price_usd': price,
                                'allocation_percentage': 0
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to parse token data: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing flat portfolio structure: {e}")
        
        return balances
    
    def _get_mock_wallet_data(self, wallet_address: str) -> Dict:
        return {
            'address': wallet_address.lower(),
            'transactions': [
                {
                    'timestamp': datetime.now() - timedelta(days=1),
                    'protocol': 'uniswap',
                    'asset_in': 'ETH',
                    'asset_out': 'USDC',
                    'value_usd': 1000.0,
                    'gas_paid': 50.0,
                    'tx_hash': '0x' + '0' * 64,
                    'method': 'swap'
                }
            ],
            'balances': {
                'ETH': {
                    'symbol': 'ETH',
                    'address': '0x0000000000000000000000000000000000000000',
                    'balance': 1.5,
                    'value_usd': 3000.0,
                    'price_usd': 2000.0,
                    'allocation_percentage': 60.0
                },
                'USDC': {
                    'symbol': 'USDC',
                    'address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                    'balance': 2000.0,
                    'value_usd': 2000.0,
                    'price_usd': 1.0,
                    'allocation_percentage': 40.0
                }
            },
            'total_value_usd': 5000.0,
            'transaction_count': 1,
            'first_transaction': datetime.now() - timedelta(days=30),
            'last_transaction': datetime.now() - timedelta(days=1)
        }

class TVLDataTool(BaseDataTool):
    name = "tvl_data"
    description = "Fetches protocol TVL from DefiLlama"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, protocol: str) -> Dict[str, Any]:
        cache_key = self._cache_key("tvl", {"protocol": protocol})
        cached_data = self._get_cached(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            response = requests.get(
                f"https://api.llama.fi/protocol/{protocol}",
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            tvl_history = data.get('tvl', [])
            current_tvl = tvl_history[-1]['totalLiquidityUSD'] if tvl_history else 0
            
            tvl_30d_ago = tvl_history[-30]['totalLiquidityUSD'] if len(tvl_history) > 30 else current_tvl
            tvl_7d_ago = tvl_history[-7]['totalLiquidityUSD'] if len(tvl_history) > 7 else current_tvl
            tvl_1d_ago = tvl_history[-1]['totalLiquidityUSD'] if len(tvl_history) > 1 else current_tvl
            
            result = {
                'protocol_name': protocol,
                'tvl': current_tvl,
                'tvl_change_24h': ((current_tvl - tvl_1d_ago) / tvl_1d_ago * 100) if tvl_1d_ago else 0,
                'tvl_change_7d': ((current_tvl - tvl_7d_ago) / tvl_7d_ago * 100) if tvl_7d_ago else 0,
                'tvl_change_30d': ((current_tvl - tvl_30d_ago) / tvl_30d_ago * 100) if tvl_30d_ago else 0,
                'audit_status': data.get('audit_links') is not None,
                'time_since_launch_days': (datetime.now() - datetime.fromtimestamp(data.get('listedAt', 0))).days,
                'recent_exploits': [],
                'risk_score': self._calculate_protocol_risk(data)
            }
            
            self._set_cached(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"DefiLlama API error: {e}")
            return self._get_mock_tvl_data(protocol)
    
    def _calculate_protocol_risk(self, data: Dict) -> float:
        risk_score = 50.0
        
        if data.get('audit_links'):
            risk_score -= 10
        
        launch_timestamp = data.get('listedAt', 0)
        if launch_timestamp:
            days_since_launch = (datetime.now() - datetime.fromtimestamp(launch_timestamp)).days
            if days_since_launch > 365:
                risk_score -= 10
            elif days_since_launch < 30:
                risk_score += 20
        
        tvl = data.get('tvl', [{}])[-1].get('totalLiquidityUSD', 0) if data.get('tvl') else 0
        if tvl > 1_000_000_000:
            risk_score -= 10
        elif tvl < 10_000_000:
            risk_score += 10
        
        return max(0, min(100, risk_score))
    
    def _get_mock_tvl_data(self, protocol: str) -> Dict:
        return {
            'protocol_name': protocol,
            'tvl': 1000000000,
            'tvl_change_24h': 2.5,
            'tvl_change_7d': 5.0,
            'tvl_change_30d': 10.0,
            'audit_status': True,
            'time_since_launch_days': 365,
            'recent_exploits': [],
            'risk_score': 30.0
        }

class NansenDataTool(BaseDataTool):
    name = "nansen_wallet_data"
    description = "Fetches wallet labels and related wallets from Nansen"
    
    def _run(self, wallet_address: str) -> Dict[str, Any]:
        if not self.config.nansen_api_key:
            return self._get_mock_nansen_data(wallet_address)
        
        cache_key = self._cache_key("nansen", {"wallet": wallet_address})
        cached_data = self._get_cached(cache_key)
        
        if cached_data:
            return cached_data
        
        return self._get_mock_nansen_data(wallet_address)
    
    def _get_mock_nansen_data(self, wallet_address: str) -> Dict:
        return {
            'wallet_address': wallet_address,
            'labels': ['Smart Money', 'DeFi User'],
            'related_wallets': [
                '0x' + '1' * 40,
                '0x' + '2' * 40,
                '0x' + '3' * 40
            ],
            'wallet_age_days': 365,
            'total_gas_spent': 1000.0,
            'protocols_used': ['uniswap', 'aave', 'compound']
        }

class SentimentDataTool(BaseDataTool):
    name = "sentiment_data"
    description = "Fetches social sentiment for tokens"
    
    def _run(self, token: str) -> Dict[str, Any]:
        cache_key = self._cache_key("sentiment", {"token": token})
        cached_data = self._get_cached(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            response = requests.get(
                "https://api.alternative.me/fng/",
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'token': token,
                'fear_greed_index': int(data['data'][0]['value']),
                'fear_greed_label': data['data'][0]['value_classification'],
                'timestamp': datetime.now().isoformat(),
                'social_volume': 1000,
                'social_sentiment': 0.6,
                'mentions_24h': 500,
                'sentiment_change_24h': 5.0
            }
            
            self._set_cached(cache_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sentiment API error: {e}")
            return self._get_mock_sentiment_data(token)
    
    def _get_mock_sentiment_data(self, token: str) -> Dict:
        return {
            'token': token,
            'fear_greed_index': 50,
            'fear_greed_label': 'Neutral',
            'timestamp': datetime.now().isoformat(),
            'social_volume': 1000,
            'social_sentiment': 0.6,
            'mentions_24h': 500,
            'sentiment_change_24h': 0.0
        }

class MarketDataTool(BaseDataTool):
    name = "market_data"
    description = "Fetches current market data"
    
    def _run(self, _: str = "") -> Dict[str, Any]:
        cache_key = self._cache_key("market", {"type": "global"})
        cached_data = self._get_cached(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            
            data = response.json()['data']
            
            result = {
                'timestamp': datetime.now(),
                'btc_price': 50000.0,
                'eth_price': 3000.0,
                'gas_price': 50.0,
                'fear_greed_index': 50,
                'market_cap_total': data.get('total_market_cap', {}).get('usd', 2000000000000),
                'volume_24h': data.get('total_volume', {}).get('usd', 100000000000),
                'dominance_btc': data.get('market_cap_percentage', {}).get('btc', 40),
                'dominance_eth': data.get('market_cap_percentage', {}).get('eth', 20)
            }
            
            self._set_cached(cache_key, result, ttl=300)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Market data API error: {e}")
            return {
                'timestamp': datetime.now(),
                'btc_price': 50000.0,
                'eth_price': 3000.0,
                'gas_price': 50.0,
                'fear_greed_index': 50,
                'market_cap_total': 2000000000000,
                'volume_24h': 100000000000,
                'dominance_btc': 40,
                'dominance_eth': 20
            }